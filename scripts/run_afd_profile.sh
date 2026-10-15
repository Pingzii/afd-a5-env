#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end AFD NPU profiling run, driven entirely from the afd-a5-env
# checkout. Nothing is written into $HOME: serve logs go to /tmp, traces go to
# the AFD_NPU_*_PROFILER_DIR the serve scripts set (/tmp/profile/{attn,ffn}).
#
# Usage (from the repo root, or via scripts/run_afd_profile.sh):
#   bash scripts/run_afd_profile.sh                 # full run
#   bash scripts/run_afd_profile.sh -n 128          # a second batch-size point
#   bash scripts/run_afd_profile.sh --no-restart    # servers already running
#   bash scripts/run_afd_profile.sh --report-only   # parse what is on disk
#
# Options:
#   -n N            concurrency                      (default 32)
#   -t TOKENS       max_tokens per request           (default 512)
#   --startup-wait S   seconds allowed per role to become healthy (default 600)
#   --shutdown-wait S  seconds allowed for graceful exit         (default 120)
#   --keep-servers  do not stop the servers at the end
#
# Requires: bash, curl, pgrep/pkill, python3.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE_F="$REPO_DIR/serve_scripts/afd_serve_F.sh"
SERVE_A="$REPO_DIR/serve_scripts/afd_serve_A.sh"
COLLECT="$REPO_DIR/scripts/collect_afd_profile.sh"

ATTN_API="127.0.0.1:18000"
FFN_API="127.0.0.1:18010"
LOG_F="/tmp/serve_F.log"
LOG_A="/tmp/serve_A.log"

N=32
TOKENS=512
STARTUP_WAIT=600
SHUTDOWN_WAIT=120
DO_RESTART=1
DO_LOAD=1
DO_REPORT=1
KEEP_SERVERS=0

while [ $# -gt 0 ]; do
  case "$1" in
    -n) N="$2"; shift ;;
    -t) TOKENS="$2"; shift ;;
    --startup-wait)  STARTUP_WAIT="$2"; shift ;;
    --shutdown-wait) SHUTDOWN_WAIT="$2"; shift ;;
    --no-restart)    DO_RESTART=0 ;;
    --report-only)   DO_RESTART=0; DO_LOAD=0; DO_REPORT=1 ;;
    --keep-servers)  KEEP_SERVERS=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
warn() { printf '  !! %s\n' "$*" >&2; }
note() { printf '  %s\n' "$*"; }

running_server_count() { pgrep -fc 'vllm serve' 2>/dev/null || echo 0; }

wait_health() {   # wait_health <host:port> <timeout_s> <label>
  local api="$1" timeout="$2" label="$3" waited=0 code
  while [ "$waited" -lt "$timeout" ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://${api}/health" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then
      note "$label healthy after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
    if [ $((waited % 60)) -eq 0 ]; then note "$label still starting (${waited}s, last http ${code})"; fi
  done
  warn "$label did not become healthy within ${timeout}s (last http ${code})"
  return 1
}

stop_servers() {  # graceful: the profiler exports in runner shutdown()
  local waited=0
  if [ "$(running_server_count)" -eq 0 ]; then
    note "no vllm serve process running"
    return 0
  fi
  note "SIGTERM (never -9: the trace is written in profiler.stop())"
  pkill -TERM -f 'vllm serve' 2>/dev/null || true
  while [ "$waited" -lt "$SHUTDOWN_WAIT" ]; do
    [ "$(running_server_count)" -eq 0 ] && { note "all servers exited after ${waited}s"; return 0; }
    sleep 5
    waited=$((waited + 5))
  done
  warn "servers still alive after ${SHUTDOWN_WAIT}s:"
  pgrep -af 'vllm serve' | sed 's/^/     /' >&2
  return 1
}

# --------------------------------------------------------------------------
say "0. preflight"
for f in "$SERVE_F" "$SERVE_A" "$COLLECT"; do
  if [ -r "$f" ]; then note "ok   $f"; else warn "missing $f"; exit 1; fi
done
if grep -q '^[[:space:]]*--profiler-config' "$SERVE_A" "$SERVE_F"; then
  warn "a serve script still passes --profiler-config as an argument;"
  warn "that enables vLLM's profiler alongside AFD's. Comment it out."
fi
note "repo      : $REPO_DIR"
note "concurrency: $N, max_tokens: $TOKENS"
note "traces     : /tmp/profile/attn and /tmp/profile/ffn"
note "serve logs : $LOG_A and $LOG_F"

if [ "$DO_REPORT" -eq 1 ] && [ "$DO_LOAD" -eq 0 ] && [ "$DO_RESTART" -eq 0 ]; then
  say "report only"
  bash "$COLLECT" --report
  exit $?
fi

# --------------------------------------------------------------------------
if [ "$DO_RESTART" -eq 1 ]; then
  say "1. stopping any running servers"
  stop_servers || true
  sleep 5

  say "2. starting the FFN role"
  : > "$LOG_F"
  nohup bash "$SERVE_F" > "$LOG_F" 2>&1 &
  note "pid $!  log $LOG_F"
  wait_health "$FFN_API" "$STARTUP_WAIT" "FFN" || { warn "see $LOG_F"; exit 1; }

  say "3. starting the Attention role"
  : > "$LOG_A"
  nohup bash "$SERVE_A" > "$LOG_A" 2>&1 &
  note "pid $!  log $LOG_A"
  wait_health "$ATTN_API" "$STARTUP_WAIT" "Attention" || { warn "see $LOG_A"; exit 1; }
else
  say "1-3. restart skipped (--no-restart)"
  wait_health "$ATTN_API" 30 "Attention" || { warn "Attention API not reachable"; exit 1; }
fi

# --------------------------------------------------------------------------
# 4. Verify the profiler env actually reached the processes. This is the
#    reliable check: the "AFD NPU ... profiler enabled" log line never appears,
#    because vLLM configures only a logger named "vllm" (propagate=False) and
#    adds no root handler, so afd_plugin.* INFO records are dropped.
say "4. verifying AFD profiler env in the running processes"
seen_attn=0
seen_ffn=0
for p in $(pgrep -f 'vllm serve'); do
  envs=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null || true)
  if printf '%s\n' "$envs" | grep -q '^AFD_NPU_FFN_PROFILER_ENABLE='; then
    seen_ffn=$((seen_ffn + 1))
    printf '%s\n' "$envs" | grep '^AFD_NPU_FFN_PROFILER_' | sed "s/^/     [pid $p] /"
  fi
  if printf '%s\n' "$envs" | grep -q '^AFD_NPU_ATTENTION_PROFILER_ENABLE='; then
    seen_attn=$((seen_attn + 1))
    printf '%s\n' "$envs" | grep '^AFD_NPU_ATTENTION_PROFILER_' | sed "s/^/     [pid $p] /"
  fi
done
if [ "$seen_ffn" -eq 0 ] || [ "$seen_attn" -eq 0 ]; then
  warn "AFD profiler env not found (attention pids=$seen_attn ffn pids=$seen_ffn)"
  warn "the servers were probably started without the serve scripts in"
  warn "serve_scripts/. Restart through this script."
  [ "$DO_LOAD" -eq 1 ] && exit 1
else
  note "ok: profile env present on $seen_attn attention pid(s) and $seen_ffn ffn pid(s)"
fi

# --------------------------------------------------------------------------
if [ "$DO_LOAD" -eq 1 ]; then
  say "5. load (${N} concurrent x ${TOKENS} tokens, ignore_eos)"
  note "curl does not toggle the AFD profiler; it only produces the model"
  note "runner steps that advance the profiler schedule."
  REQ="/tmp/afd_prof_req.json"
  cat > "$REQ" <<JSON
{
  "model": "deepseek-v4",
  "messages": [
    {"role": "user", "content": "Write a detailed technical explanation of how transformer attention works, covering queries, keys, values, and the scaling factor."}
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": ${TOKENS},
  "ignore_eos": true
}
JSON
  t0=$(date +%s)
  for _ in $(seq 1 "$N"); do
    curl -s -o /dev/null "http://${ATTN_API}/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @"$REQ" &
  done
  wait
  note "load finished in $(( $(date +%s) - t0 ))s"
fi

# --------------------------------------------------------------------------
if [ "$KEEP_SERVERS" -eq 0 ]; then
  say "6. graceful shutdown (this is what makes the traces land)"
  stop_servers || true
else
  say "6. shutdown skipped (--keep-servers); traces will only appear after a"
  note "graceful stop, so run this again and let it stop them."
  exit 0
fi

# --------------------------------------------------------------------------
say "7. report"
bash "$COLLECT" --report
rc=$?

say "done"
note "traces: /tmp/profile/attn and /tmp/profile/ffn"
note "if the report could not identify the CSV columns, send it the header"
note "line and one data row it printed; the column names get pinned."
exit $rc
