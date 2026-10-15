#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# AFD NPU profiling collector for the sync CAMP2P A2E/E2A question.
#
# Purpose: after A and F are started WITH the AFD profiler env vars enabled,
# generate a decode-heavy load, then extract the per-kernel Wall Duration for
# the A2e / E2a operators on BOTH roles.
#
# Usage:
#   bash scripts/collect_afd_profile.sh --load          # fire the load only
#   bash scripts/collect_afd_profile.sh --report        # parse what is on disk
#   bash scripts/collect_afd_profile.sh                 # load, then report
#
# Options:
#   -n N          concurrency (default 32; use 32 and 128 for two points)
#   -t TOKENS     max_tokens per request (default 512)
#   -A HOST:PORT  Attention API (default 127.0.0.1:18000)
#   --attn-dir D  attention profiler dir (default /tmp/profile/attn)
#   --ffn-dir  D  ffn profiler dir       (default /tmp/profile/ffn)
#   --model  M    served model name      (default deepseek-v4)
#   --log     L   serve log(s) for the profiler-enabled check. May be given
#                 more than once. Defaults to
#                 "/tmp/serve_F.log /tmp/serve_A.log /home/s00988495/afd_serve.log"
#                 (only the paths that exist are read)
#
# Requires: bash, curl, python3. Nothing else.

set -u

N=32
TOKENS=512
ATTN_API="127.0.0.1:18000"
ATTN_DIR="/tmp/profile/attn"
FFN_DIR="/tmp/profile/ffn"
MODEL="deepseek-v4"
SERVE_LOG="/tmp/serve_F.log /tmp/serve_A.log /home/s00988495/afd_serve.log"
DO_LOAD=1
DO_REPORT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --load)   DO_REPORT=0 ;;
    --report) DO_LOAD=0 ;;
    -n)       N="$2"; shift ;;
    -t)       TOKENS="$2"; shift ;;
    -A)       ATTN_API="$2"; shift ;;
    --attn-dir) ATTN_DIR="$2"; shift ;;
    --ffn-dir)  FFN_DIR="$2"; shift ;;
    --model)  MODEL="$2"; shift ;;
    --log)    if [ "${LOG_SET:-0}" -eq 0 ]; then SERVE_LOG="$2"; LOG_SET=1
              else SERVE_LOG="$SERVE_LOG $2"; fi; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

REQ_JSON="${TMPDIR:-/tmp}/afd_prof_req.json"

say() { printf '\n=== %s ===\n' "$*"; }

# --------------------------------------------------------------------------
# 0. pre-check: is AFD's profiler actually enabled?
#
#    Two DIFFERENT profilers can appear in the log. Do not confuse them:
#
#      vLLM's own   : "Torch profiler enabled. AsyncLLM CPU traces will be
#                      collected under <dir>"   (vllm/v1/engine/async_llm.py)
#      AFD's        : "AFD NPU <role> profiler enabled. Traces will be saved
#                      to: <dir>"               (afd_plugin/compat/npu/profiler.py)
#
#    Only AFD's is wired into the NPU model runners, so only AFD's collects the
#    A2e/E2a kernel timings from the runner schedule. AFD's profiler is created
#    in the runner __init__ from AFD_NPU_{ATTENTION,FFN}_PROFILER_* variables,
#    so it can only be turned on BEFORE the process starts. This check saves a
#    wasted load.
# --------------------------------------------------------------------------
say "0. AFD profiler pre-check"

found_log=0
n_attn=0
n_ffn=0
n_vllm=0
for lg in $SERVE_LOG; do
  [ -r "$lg" ] || continue
  found_log=1
  echo "  log: $lg"
  grep 'AFD NPU .*profiler enabled' "$lg" 2>/dev/null | sed 's/^/    /'
  n_attn=$((n_attn + $(grep -c 'AFD NPU attention profiler enabled' "$lg" 2>/dev/null || true)))
  n_ffn=$((n_ffn + $(grep -c 'AFD NPU ffn profiler enabled' "$lg" 2>/dev/null || true)))
  n_vllm=$((n_vllm + $(grep -c 'Torch profiler enabled' "$lg" 2>/dev/null || true)))
done

if [ "$found_log" -eq 0 ]; then
  echo "  (none of these logs is readable: $SERVE_LOG)"
  echo "  pass the right ones with:  --log /path/serve_F.log --log /path/serve_A.log"
fi

if [ "$n_vllm" -gt 0 ]; then
  cat >&2 <<'WARN'
  NOTE: vLLM's OWN profiler is enabled (lines containing "Torch profiler
        enabled. AsyncLLM CPU traces ..."). That is a DIFFERENT profiler and
        it is not what this script reads. AFD's profiler is the one wired into
        the NPU model runners.
WARN
fi

if [ "$n_attn" -eq 0 ] || [ "$n_ffn" -eq 0 ]; then
  cat >&2 <<'HOWTO'
  !! AFD's profiler is NOT enabled. Expected these lines in the serve logs:

       AFD NPU attention profiler enabled. Traces will be saved to: /tmp/profile/attn
       AFD NPU ffn profiler enabled. Traces will be saved to: /tmp/profile/ffn

     Add this to the serve script BEFORE the `exec vllm serve` line, for BOTH
     roles, then restart F and A:

       export AFD_NPU_FFN_PROFILER_ENABLE=1
       export AFD_NPU_FFN_PROFILER_DIR=/tmp/profile/ffn
       export AFD_NPU_FFN_PROFILER_WAIT=20          # code default is 2500
       export AFD_NPU_FFN_PROFILER_WARMUP=2
       export AFD_NPU_FFN_PROFILER_ACTIVE=20        # must match on both roles
       export AFD_NPU_FFN_PROFILER_REPEAT=1
       export AFD_NPU_FFN_PROFILER_SKIP_FIRST=0
       export AFD_NPU_FFN_PROFILER_WITH_STACK=0

     and the same eight with AFD_NPU_ATTENTION_PROFILER_* / DIR=/tmp/profile/attn
     on the Attention role.

     --report still works on previously collected data.
HOWTO
else
  echo "  OK: both AFD profilers are enabled (attention=$n_attn ffn=$n_ffn)."
fi

# --------------------------------------------------------------------------
# 1. load
#    curl does NOT toggle the AFD profiler -- it only produces the model
#    runner steps that advance the profiler's schedule.
# --------------------------------------------------------------------------
if [ "$DO_LOAD" -eq 1 ]; then
  say "1. health check"
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://${ATTN_API}/health" || echo 000)
  echo "  GET /health -> $code"
  if [ "$code" != "200" ]; then
    echo "  !! Attention API not healthy; aborting load" >&2
    exit 1
  fi

  cat > "$REQ_JSON" <<JSON
{
  "model": "${MODEL}",
  "messages": [
    {"role": "user", "content": "Write a detailed technical explanation of how transformer attention works, covering queries, keys, values, and the scaling factor."}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": ${TOKENS},
  "ignore_eos": true
}
JSON

  say "2. firing ${N} concurrent requests (max_tokens=${TOKENS}, ignore_eos=true)"
  t0=$(date +%s)
  for i in $(seq 1 "$N"); do
    curl -s -o /dev/null "http://${ATTN_API}/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @"$REQ_JSON" &
  done
  wait
  t1=$(date +%s)
  echo "  done in $((t1 - t0))s"

  say "3. shutting down gracefully so the profiler can export"
  echo "  Sending SIGTERM (NOT -9). The trace is written in profiler.stop(),"
  echo "  which runs from the runner's shutdown(). SIGKILL loses it."
  echo "  >>> Do this yourself and wait for the processes to exit:"
  echo "        pkill -TERM -f 'vllm serve' ; sleep 60"
  echo "        ps aux | grep -c '[v]llm serve'   # expect 0"
fi

# --------------------------------------------------------------------------
# 4. locate the profiler output dirs
# --------------------------------------------------------------------------
if [ "$DO_REPORT" -eq 0 ]; then
  exit 0
fi

say "4. locating profiler output"

for d in "$ATTN_DIR" "$FFN_DIR"; do
  if [ -d "$d" ]; then
    echo "  $d:"
    find "$d" -maxdepth 2 -name '*_ascend_pt' -type d 2>/dev/null | sed 's/^/    /'
  else
    echo "  $d: MISSING"
  fi
done

# --------------------------------------------------------------------------
# 5. ensure ASCEND_PROFILER_OUTPUT exists, then extract the key numbers.
#    Raw profiler data is binary; ASCEND_PROFILER_OUTPUT/ is the readable
#    translation. analyse() generates it when it is absent.
# --------------------------------------------------------------------------
say "5. ensuring analysis output + extracting A2e / E2a"

python3 - "$ATTN_DIR" "$FFN_DIR" <<'PY'
import csv
import sys
from pathlib import Path

roots = [Path(a) for a in sys.argv[1:3]]
WANT = ("a2e", "e2a", "gmm", "matmul", "swiglu", "dispatch", "combine")
DUR_KEYS = ("Wall Duration", "Duration", "wall_duration", "duration",
            "Total Time", "total_time")

def ensure_analysis(d: Path) -> None:
    out = d / "ASCEND_PROFILER_OUTPUT"
    if out.is_dir():
        return
    try:
        from torch_npu.profiler.profiler import analyse  # private path
    except Exception as exc:  # pragma: no cover
        print(f"  [{d.name}] cannot import analyse(): {exc}")
        return
    print(f"  [{d.name}] ASCEND_PROFILER_OUTPUT missing -> analyse()")
    try:
        analyse(str(d))
    except Exception as exc:  # pragma: no cover
        print(f"  [{d.name}] analyse() failed: {exc}")

def parse_duration(text: str) -> float:
    """Best-effort: return a float in the value's own unit (usually us or ns)."""
    t = str(text).strip().replace(",", "")
    for unit, scale in (("ns", 1e-3), ("us", 1.0), ("ms", 1e3), ("s", 1e6)):
        if t.endswith(unit):
            try:
                return float(t[: -len(unit)]) * scale
            except ValueError:
                return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")

for root in roots:
    print(f"\n--- {root} ---")
    if not root.is_dir():
        print("  missing")
        continue
    for d in sorted(p for p in root.rglob("*_ascend_pt") if p.is_dir()):
        ensure_analysis(d)
        out = d / "ASCEND_PROFILER_OUTPUT"
        csvs = sorted(out.glob("*kernel*.csv")) if out.is_dir() else []
        if not csvs:
            csvs = sorted(out.glob("*.csv")) if out.is_dir() else []
        if not csvs:
            print(f"  [{d.name}] no kernel csv under {out}")
            continue
        src = csvs[0]
        print(f"  [{d.name}] {src.name}")
        with src.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            print(f"    columns: {cols[:14]}{' ...' if len(cols) > 14 else ''}")

            name_col = next((c for c in cols if str(c).strip().lower() in
                             ("title", "kernel_name", "kernel name", "name")), None)
            dur_col = next((c for c in cols if str(c).strip().lower() in
                            tuple(k.lower() for k in DUR_KEYS)), None)
            if name_col is None or dur_col is None:
                print("    !! could not identify name/duration columns;")
                print("       send the header line above and the first data row.")
                for row in list(reader)[:2]:
                    print(f"       {row}")
                continue

            agg: dict[str, list[float]] = {}
            for row in reader:
                n = str(row.get(name_col, ""))
                low = n.lower()
                if not any(w in low for w in WANT):
                    continue
                agg.setdefault(n, []).append(parse_duration(row.get(dur_col, "")))

            if not agg:
                print("    no rows matched a2e/e2a/gmm/matmul/swiglu")
                continue
            print(f"    {'kernel':<34} {'n':>5} {'total':>12} {'mean':>10} {'max':>10}  (unit from column)")
            for name in sorted(agg, key=lambda k: -sum(agg[k])):
                v = [x for x in agg[name] if x == x]
                if not v:
                    continue
                print(f"    {name[:34]:<34} {len(v):>5} {sum(v):>12.1f} "
                      f"{sum(v)/len(v):>10.1f} {max(v):>10.1f}")

print("\nNote: A2e on the Attention role is a LOCAL window write.")
print("      A2E's cross-rank cost is the FFN role's A2e (the pull read).")
print("      Compare the two E2a rows to tell 'waiting for FFN' from 'slow read'.")
PY

say "done"
echo "Next: report the FFN-side A2e duration and the FFN-side GMM duration."
