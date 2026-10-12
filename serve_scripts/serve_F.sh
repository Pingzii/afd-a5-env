#!/usr/bin/env bash
# AFD FFN 角色启动脚本 - 2A2F topology
set -euo pipefail

MODEL_PATH="/mnt/weight/A5-weights/DeepSeek-V4-Flash"
SERVED_MODEL_NAME="DeepSeek-V4-Flash"
FFN_DEVICES="2,3"
ATTN_DEVICES="0,1"
ATTN_PORT="18000"
FFN_PORT="18010"
AFD_HOST="127.0.0.1"
AFD_PORT="6239"
MAX_MODEL_LEN=3000
MAX_NUM_SEQS=256
ATTN_RANKS=2
FFN_RANKS=2

export ASCEND_RT_VISIBLE_DEVICES="$FFN_DEVICES"
export VLLM_PLUGINS="ascend,afd"
export PYTHONUNBUFFERED="1"
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

AFD_CONFIG="$(printf '{"afd":{"role":"ffn","connector":"CAMP2pAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' \
    "$AFD_HOST" "$AFD_PORT" "$ATTN_RANKS" "$FFN_RANKS")"

exec vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --compilation-config '{"cudagraph_capture_sizes":[32,64,128,256], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --host 127.0.0.1 \
    --port "$FFN_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --trust-remote-code \
    --additional-config "$AFD_CONFIG"
