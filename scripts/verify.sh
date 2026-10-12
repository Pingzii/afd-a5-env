#!/usr/bin/env bash
# Green-zone side of the blue/green loop: apply a patch stack, verify it in three
# layers, revert.
#
# Layer 1  the stack applies to this checkout      (git apply --check)
# Layer 2 the service comes up                     (startup log / endpoint ready)
# Layer 3 the output equals the reference          (greedy decode, text compare)
#
# Layer 3 is the one that gets skipped, and it is the only one that catches a
# silent wrong answer -- a change can apply cleanly and serve happily while
# producing garbage. A reference is mandatory: without one there is no evidence.
#
# ASCII only, no jq dependency (python3 is assumed; the preflight already needs it).
#
# Usage
#   verify.sh check   -r vllm-ascend -n dsv4-a5-cache-dtype
#   verify.sh apply   -r vllm-ascend -n dsv4-a5-cache-dtype
#   verify.sh revert  -r vllm-ascend -n dsv4-a5-cache-dtype
#   verify.sh status  [-r vllm-ascend]
#   verify.sh baseline -o ref.txt  [--url http://127.0.0.1:18000] [--prompt "..."] [--tokens 10]
#   verify.sh compare  -b ref.txt  [--url ...] [--prompt ...] [--tokens 10] [--restart "cmd"]
#
# Exit codes: 0 pass, 1 fail (differs / not clean / service not ready), 2 misuse.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="$(cd "$SELF_DIR/.." && pwd)"
WORKSPACE="$(cd "$ENV_ROOT/.." && pwd)"

URL="http://127.0.0.1:18000"
PROMPT="Who are you?"
TOKENS=10
REPO=""
NAME=""
OUT=""
BASE=""
RESTART=""

die() { echo "error: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        -r|--repo)    REPO="$2"; shift 2 ;;
        -n|--name)    NAME="$2"; shift 2 ;;
        -o|--out)     OUT="$2"; shift 2 ;;
        -b|--base)    BASE="$2"; shift 2 ;;
        --url)        URL="$2"; shift 2 ;;
        --prompt)     PROMPT="$2"; shift 2 ;;
        --tokens)     TOKENS="$2"; shift 2 ;;
        --restart)    RESTART="$2"; shift 2 ;;
        -*)           die "unknown option: $1" ;;
        *)            ACTION="$1"; shift ;;
    esac
done

[ -n "${ACTION:-}" ] || die "usage: verify.sh <check|apply|revert|status|baseline|compare> [options]"

patch_dir() {
    [ -n "$REPO" ] || die "-r/--repo is required"
    [ -n "$NAME" ] || die "-n/--name is required"
    echo "$ENV_ROOT/patches/$REPO/$NAME"
}

src_repo() {
    [ -n "$REPO" ] || die "-r/--repo is required"
    echo "$WORKSPACE/$REPO"
}

load_stack() {
    local d; d="$(patch_dir)"
    [ -d "$d" ] || die "no such patch set: $d"
    STACK=()
    local f
    while IFS= read -r f; do
        [ -n "$f" ] && STACK+=("$f")
    done < <(find "$d" -maxdepth 1 -name '*.patch' ! -name 'combined.patch' | sort)
    [ "${#STACK[@]}" -gt 0 ] || die "no NNNN-*.patch files in $d"
}

worktree_dirty() {
    git -C "$(src_repo)" status --porcelain | wc -l | tr -d ' '
}

show_status() {
    local p; p="$(src_repo)"
    echo "  repo:    $REPO"
    echo "  HEAD:    $(git -C "$p" rev-parse --short HEAD)"
    echo "  branch:  $(git -C "$p" rev-parse --abbrev-ref HEAD)"
    local n; n="$(worktree_dirty)"
    echo "  dirty:   $n"
    git -C "$p" status --porcelain | sed 's/^/    /'
}

do_check() {
    local p; p="$(src_repo)"
    [ "$(worktree_dirty)" -eq 0 ] || die "worktree is not clean -- revert first"
    load_stack
    local bad=0 f
    for f in "${STACK[@]}"; do
        if git -C "$p" apply --check "$f" 2>/dev/null; then
            echo "  [ok]     $(basename "$f")"
        else
            echo "  [FAILED] $(basename "$f")"
            git -C "$p" apply --check "$f" 2>&1 | sed 's/^/           /'
            bad=1
        fi
    done
    [ "$bad" -eq 0 ] || { echo "  the stack does not apply to this base"; exit 1; }
    echo "  check passed"
}

do_apply() {
    local p; p="$(src_repo)"
    [ "$(worktree_dirty)" -eq 0 ] || die "worktree is not clean -- revert first"
    load_stack
    local f
    for f in "${STACK[@]}"; do
        git -C "$p" apply "$f" || die "apply failed: $(basename "$f")"
        echo "  applied $(basename "$f")"
    done
    echo "  resulting worktree:"
    git -C "$p" status --porcelain | sed 's/^/    /'
    echo
    echo "  next: restart, then 'verify.sh compare -b <ref>' ; finish with 'verify.sh revert'"
}

do_revert() {
    local p; p="$(src_repo)"
    load_stack
    local i f rel
    for (( i=${#STACK[@]}-1; i>=0; i-- )); do
        f="${STACK[$i]}"
        if git -C "$p" apply -R "$f" 2>/dev/null; then
            echo "  reverted $(basename "$f")"
        else
            echo "  warning: reverse apply failed (already reverted?): $(basename "$f")" >&2
        fi
        # 'apply -R' restores modified files but leaves CREATED files behind.
        while IFS= read -r rel; do
            [ -n "$rel" ] || continue
            if [ -e "$p/$rel" ]; then rm -f "$p/$rel"; echo "  removed added file $rel"; fi
        done < <(awk '/^diff --git /{f=""} /^\+\+\+ b\//{f=substr($0,7)} /^new file mode/{if(f)print f}' "$f")
    done
    if [ "$(worktree_dirty)" -ne 0 ]; then
        echo "  worktree is NOT clean -- resolve by hand before the next experiment" >&2
        git -C "$p" status --porcelain | sed 's/^/    /' >&2
        exit 1
    fi
    echo "  revert complete, worktree clean"
}

call_model() {
    curl -fsS "$URL/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\": $(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'), \
             \"max_tokens\": $TOKENS, \"temperature\": 0}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"], end="")'
}

wait_ready() {
    local i
    for i in $(seq 1 120); do
        if curl -fsS -o /dev/null "$URL/v1/models" 2>/dev/null; then
            echo "  service ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  service did not become ready within 120s" >&2
    return 1
}

do_baseline() {
    [ -n "$OUT" ] || die "-o/--out is required"
    if [ -n "$RESTART" ]; then echo "  restarting: $RESTART"; eval "$RESTART"; wait_ready; fi
    local text; text="$(call_model)"
    printf '%s\n' "$text" > "$OUT"
    echo "  captured reference -> $OUT"
    echo "  prompt=$PROMPT tokens=$TOKENS temperature=0"
    printf '  text: %s\n' "$text"
}

do_compare() {
    [ -n "$BASE" ] || die "-b/--base is required"
    [ -f "$BASE" ] || die "no such reference: $BASE"
    if [ -n "$RESTART" ]; then echo "  restarting: $RESTART"; eval "$RESTART"; wait_ready; fi
    local now; now="$(call_model)"
    local ref; ref="$(cat "$BASE")"
    if [ "$now" = "$ref" ]; then
        echo "  layer 3 PASS: output is byte-identical to the reference"
        printf '  text: %s\n' "$now"
        exit 0
    fi
    echo "  layer 3 FAIL: output differs from the reference" >&2
    echo "    reference: $ref" >&2
    echo "    now:       $now" >&2
    echo "  a change that applies and serves but changes the output is a silent wrong answer." >&2
    exit 1
}

case "$ACTION" in
    check)    do_check ;;
    apply)    do_apply ;;
    revert)   do_revert ;;
    status)   show_status ;;
    baseline) do_baseline ;;
    compare)  do_compare ;;
    *)        die "unknown action: $ACTION" ;;
esac
