#!/usr/bin/env bash
# AFD FFN role -- Ascend 950 (A5), CAMP2pAFDConnector, 4A4F.
#
# Topology : Attention DP4/TP1 on devices 0-3, FFN DP4/TP1 on devices 4-7.
# Start    : start this script first, then afd_serve_A.sh.
# Stop     : pkill -TERM -f 'vllm serve'     # SIGTERM, never -9
#
# Profiling --------------------------------------------------------------
# This script enables AFD's own NPU profiler, which is owned by
# afd_plugin/compat/npu/profiler.py and wired into both NPU model runners.
# The eight AFD_NPU_<ROLE>_PROFILER_* variables are read ONCE, in the runner
# __init__, so they must be exported before `exec vllm serve`.
#
# --profiler-config is deliberately NOT passed. It enables vLLM's separate
# torch profiler in the SAME process, and two torch_npu profilers contend.
# Use one or the other, not both.
#
# Two things that are easy to get wrong:
#   * AFD_NPU_*_PROFILER_SKIP_FIRST defaults to 1500. A short run must set 0.
#   * AFD_NPU_*_PROFILER_ACTIVE defaults to 10 (attention) / 20 (ffn). Set the
#     same value on both roles so the capture windows line up.
#
# The FFN role is driven by the Attention role through the connector, so never
# send client requests to port 18010 -- that port only answers health and
# metadata queries.
#
# The "AFD NPU ... profiler enabled" log line will NOT appear in the serve
# log. vLLM configures only a logger named "vllm" (with propagate=False) and
# never adds a root handler, so afd_plugin.* INFO records are dropped. Verify
# the profiler instead with:
#     tr '\0' '\n' < /proc/<pid>/environ | grep AFD_NPU_
# and after a graceful shutdown with:
#     ls -la /tmp/profile/ffn/
#
# A2E note: on this role, A2e performs the cross-rank pull read of the
# Attention window, which is where the A2E transfer cost actually lands. The
# Attention role's own A2e is only a local window write.

set -euo pipefail

# ---- configuration -----------------------------------------------------
MODEL_PATH="/mnt/share/weight/DeepSeek-V4-Flash"
SERVED_MODEL_NAME="deepseek-v4"

FFN_DEVICES="4,5,6,7"            # FFN devices
ATTN_DEVICES="0,1,2,3"           # Attention devices (kept for log clarity)
FFN_PORT="18010"
ATTN_PORT="18000"
AFD_HOST="127.0.0.1"
AFD_PORT="6239"
MAX_MODEL_LEN="8192"
# No log path and no stdout redirect here: everything this profile run produces
# stays under /tmp or under the afd-a5-env checkout. Run the script through
# scripts/run_afd_profile.sh, or redirect yourself:
#     nohup bash serve_scripts/afd_serve_F.sh > /tmp/serve_F.log 2>&1 &

export ASCEND_RT_VISIBLE_DEVICES="$FFN_DEVICES"
export VLLM_PLUGINS="ascend,afd"
export PYTHONUNBUFFERED="1"
export VLLM_USE_V1="1"
export HCCL_BUFFSIZE="800"
# 950 pitfall: AIV vector kernels (all_gather and friends) trip vector core
# error 507035. Leaving HCCL_OP_EXPANSION_MODE unset keeps the default iamg
# tiling and avoids the AIV path; both DIRECT and AIV trigger it.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"

# ---- AFD NPU profiler (FFN role) ---------------------------------------
export AFD_NPU_FFN_PROFILER_ENABLE="1"
export AFD_NPU_FFN_PROFILER_DIR="/tmp/profile/ffn"
export AFD_NPU_FFN_PROFILER_WAIT="20"
export AFD_NPU_FFN_PROFILER_WARMUP="2"
export AFD_NPU_FFN_PROFILER_ACTIVE="20"              # match the Attention role
export AFD_NPU_FFN_PROFILER_REPEAT="1"
export AFD_NPU_FFN_PROFILER_SKIP_FIRST="0"           # default is 1500
export AFD_NPU_FFN_PROFILER_WITH_STACK="0"

# ---- vLLM's own torch profiler -- keep disabled while AFD's is on -------
#    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \

exec vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --compilation-config '{"cudagraph_capture_sizes":[32,64,128,256], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --host 127.0.0.1 \
    --port "$FFN_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --additional-config "{\"afd\":{\"role\":\"ffn\",\"connector\":\"CAMP2pAFDConnector\",\"host\":\"$AFD_HOST\",\"port\":$AFD_PORT,\"num_attention_ranks\":4,\"num_ffn_ranks\":4}}"
