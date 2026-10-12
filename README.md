# AFD A5 环境快照

Ascend A5 上 Attention/FFN 分离部署（AFD）的**可复现环境快照**：四个仓的源码 + 启动脚本 + 版本清单。

## 版本清单

| 仓 | commit | 版本 | 上游 |
|---|---|---|---|
| `afd-plugin` | `bf5bd28` | — | https://github.com/Pingzii/afd-plugin |
| `vllm` | `ced6857afa` | `v0.30.0` | https://github.com/Pingzii/vllm |
| `vllm-ascend` | `8d4409d62` | `v0.19.1rc1-2448-g8d4409d62` | https://github.com/Pingzii/vllm-ascend |
| `ops-transformer`（CANN） | `951365ce8` | `v8.5.0-beta.1-6426-g951365ce8` | https://gitcode.com/cann/ops-transformer |

**这四个 commit 是一组经过验证的配套。**

## 注意

- **本仓是快照，不含 `.git`。** 所以不能 `git diff` / `git pull`，也不能追溯历史。
  需要历史时请回到上游仓 checkout 上表对应的 commit。
- 各仓的 LICENSE 文件原样保留。它们的授权各自适用，本仓只是打包。
- 快照体积较大（约 420 MB），clone 时请留够空间。

## 目录

```
.
├── README.md
├── serve_scripts/
│   ├── serve_A.sh          # Attention 角色（2A2F）
│   └── serve_F.sh          # FFN 角色（2A2F）
├── afd-plugin/
├── vllm/
├── vllm-ascend/
└── ops-transformer/        # CANN 算子库
```

## 部署形态（2A2F）

```
单机 4 卡 · DP2 × TP1 · EP 开启
  ATTN_DEVICES=0,1   ATTN_PORT=18000
  FFN_DEVICES=2,3    FFN_PORT=18010
  connector=CAMP2pAFDConnector（同步）· num_attention_ranks=2 · num_ffn_ranks=2
```

## 模型

```
/mnt/weight/A5-weights/DeepSeek-V4-Flash
```

| 项 | 值 |
|---|---|
| `hidden_size` | 4096 |
| `n_routed_experts` | 256 |
| `num_experts_per_tok` | 6 |
| `num_hidden_layers` | 43 |
| `moe_intermediate_size` | 2048 |
| **`expert_dtype`** | **`fp4`** |
| `quantization_config` | `quant_method=fp8`、`fmt=e4m3`、`scale_fmt=ue8m0`、`weight_block_size=[128,128]` |

→ MoE 走 vllm-ascend 的 `W4A8MXFP` scheme（权重 MXFP4 E2M1、激活 MXFP8 E4M3、block 32、e8m0 scale）。
启动脚本**没有传 `--quantization`**，走的是 native FP8 路径（`AscendFp8Config`），不是 modelslim。

## 怎么起

```bash
# 两个终端，或同一终端的两个后台进程
bash serve_scripts/serve_A.sh      # Attention 角色
bash serve_scripts/serve_F.sh      # FFN 角色
```

两个脚本共用 `AFD_PORT=6239` 做控制面握手，所以**必须都起来**才能工作。
另需环境里已装好 CANN、`torch` / `torch_npu`，以及本仓的 `vllm` / `vllm-ascend` / `afd-plugin`。
