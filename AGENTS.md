# AGENTS.md — AFD A5 量化工作区

> **本文件是工作区级导航，不是子仓规范。**
> 每个子仓有自己的 `AGENTS.md`，**进仓改代码前必须读那一份**。
> 最后更新：2026-10-12

---

## 一、这个工作区在做什么

Ascend A5 上 Attention/FFN 分离部署（AFD）的 **A2E 通信量化**。

```
Attention rank                         FFN rank
  BF16 hidden_states [bs, H]
      │  ① 框架侧量化（torch_npu 的 npu_dynamic_mx_quant）
      ▼
  fp8 payload + e8m0 scale
      │  ② a2e 算子搬 payload + scale
      ▼
  ────────── HCCL window ──────────→  收到 fp8 + scale
                                          │  ③ dispatch 不重复量化（quant_mode=0）
                                          ▼  ④ GEMM1 直接用外部 scale
```

**目标收益**：A2E 通信 `H×2` → `H + H/32` 字节/token。
`H=4096` 时是 **8192 → 4224 B/token（-48.4%）**。

**不在范围**：`e2a` 不改（它下游是残差加 + RMSNorm，BF16 域，量化了要反量化，净负收益）。

---

## 二、仓的角色

| 目录 | 角色 | 能改吗 | 自己的 AGENTS.md |
|---|---|---|---|
| `afd-plugin/` | **主工作目标** | ✅ | ✅ **必读** |
| `vllm-ascend/` | Ascend 硬件插件 | ⚠️ 开发期可直接改；交付走 **该项目的 patch**：`afd-plugin/patches/vllm-ascend/` | ✅ **必读** |
| `vllm/` | 上游 vLLM | ❌ 只读参考 | ✅ |
| `ops-transformer/` | CANN 算子库 | ❌ 第三方，只读参考 | ❌ 无 |
| `serve_scripts/` | 2A2F 启动脚本 | ✅ | — |
| `archive/` | 已废弃的历史文档 | ❌ 只做参考，**别引用其结论** | — |
| `_afd_env_repo/` | 本项目私仓 `Pingzii/afd-a5-env` 的环境快照暂存 | ⚠️ 别手工改，用脚本重建 | — |

**`ops-transformer` 的用法**：不是为了改它，而是**为了核实算子声明**。
当有人（或某份文档）说"这个算子支持 X"时，去它源码里看 `op_host/*_def.cpp` 的 dtype 组合表
和 `op_tiling/` 的校验函数 —— 这是唯一能证伪的地方。

---

## 三、版本锁定（四个 commit 是一组配套）

| 仓 | commit | 版本 |
|---|---|---|
| `afd-plugin` | `bf5bd28` | — |
| `vllm` | `ced6857afa` | `v0.30.0` |
| `vllm-ascend` | `8d4409d62` | `v0.19.1rc1-2448-g8d4409d62` |
| `ops-transformer` | `951365ce8` | `v8.5.0-beta.1-6426-g951365ce8` |

> 升级任何一个之前先确认四个的兼容性。
> **0.26 → 0.30 之间 vllm-ascend 的目录结构变过**（量化方法按类型分了子目录、
> `pertoken_scale` 的注入方式变了），跨版本引用行号前务必重新核对。

---

## 四、模型与部署

**模型**：`/mnt/weight/A5-weights/DeepSeek-V4-Flash`

| 项 | 值 |
|---|---|
| `hidden_size` | 4096 |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 6 |
| `num_hidden_layers` | 43 |
| **`expert_dtype`** | **`fp4`** |
| `quantization_config` | `quant_method=fp8`、`fmt=e4m3`、`scale_fmt=ue8m0`、`weight_block_size=[128,128]` |

**部署**：单机 4 卡（`serve_scripts/`），DP2 × TP1 + EP，同步 connector（CAMP2pAFDConnector）。

---

## 五、改代码前必须知道的硬约束

这几条是代价很高的结论，别重新推导：

1. **现状量化方案是 `W4A8MXFP`** —— 权重 MXFP4(E2M1) × 激活 MXFP8(E4M3)，block 32，e8m0 scale。
   由 `config.json` 的 `quant_method: fp8`（→ `AscendFp8Config`）+ `expert_dtype: fp4` 决定。
   **启动脚本没有 `--quantization`**，走的是 native FP8 路径，不是 modelslim（模型目录里没有
   `quant_model_description.json`）。**所以"换 MXFP8 checkpoint"这个动作是多余的 —— 已经在 MXFP8 上了。**

2. **router 只能吃 FP16/FP32/BF16** —— `GateLinear` 的 `quant_config=None` 是硬编码的
   （`vllm/.../router/gate_linear.py`），`MoeGatingTopK` 的 OpDef 也没有量化类型。
   **如果 FFN 只收到 fp8，router 必须先拿到一份 BF16**（反量化，或把 router 挪到 Attention 侧）。

3. **A2E 是整条链上唯一还传 BF16 的一段** ——
   FFN 内层的 EP dispatch 已经在传量化数据（`dispatch_with_quant` 包含 `W4A8MXFP`）。

4. **A5 有两条以上的 MoE 通信路径**（ALLGATHER / MC2 / ALLTOALL），
   量化发生的位置各不相同。**DP 是动态配置的，三条都要覆盖。**
   改任何一条之前先确认当前实际走哪条。

5. **CANN 官方已有 AFD 通信算子对**，A5 支持，且原生支持 MXFP8：
   - `mc2/attention_to_ffn_v2`：`quantMode=4` → MX FP8_E4M3 + E8M0，`expertIds` 是必需输入
   - `ffn/ffn_worker_batching`：`token_dtype=4` → FLOAT8_E4M3FN，输出 e8m0 scale + `group_list`
   **它们和 AFD 自研的 a2e/e2a 是两套不同协议**（`ScheduleContext` vs HCCL window + flag），
   不是 drop-in 替换，但值得评估"自研 vs 用官方"。

---

## 六、工作流程

### 蓝绿协作（改绿区必须走这条）

绿区是**只读**的：只能 `git clone/fetch`、改本地文件、跑本地进程、写 `/tmp`。
**任何 push / upload 都是合规违规。**

**每个项目使用自己的私仓。** 本项目对应 `Pingzii/afd-a5-env`，保存配套源码快照、
版本说明、`AGENTS.md`、执行脚本、项目 patch、排错规范、踩坑记录和当前进展。

```
项目自有源码
    → 蓝区修改、验证并 push 项目私仓的精确提交
    → 绿区 fetch 该提交并按项目脚本运行

vllm-ascend 等外部仓改动
    → 蓝区按固定 base 生成 patch，随项目私仓保存
    → 绿区 fetch 项目提交，先 git apply --check，再应用和运行
```

**硬约束**（完整版见 `.agents/skills/blue-green-collab/SKILL.md`）：

```
· 动手前先记基线（git status --porcelain），事后只撤本次变更，再核对工作树为空
· 不留没有撤销路径的改动
· 一次实验只变一个变量，并说明相对上次变了什么
· 结果只回项目约定的最短结论，必要时附首错的少量上下文
· git apply --check 失败就停下，不许就地改目标文件
· 不把「未验证」写成「已完成」
```

详见 `.agents/skills/blue-green-collab/SKILL.md`。

### 写 PowerShell 脚本时：纯 ASCII

**Windows PowerShell 5.1 在没有 BOM 时按 ANSI 读 `.ps1`。** 用 UTF-8 无 BOM 写中文注释，
解析器会把字符串读坏、整个脚本连解析都过不去（表现是 `Unexpected token` / `missing the terminator`，
**一行都不会执行**）。

要么脚本纯 ASCII，要么用 `pwsh`（PowerShell 7，默认 UTF-8）执行。

### 网络：HTTPS 不通，用 SSH

这台机器上 git 的 `http.proxy` 指向 `127.0.0.1:7890`，但那个代理连不出去
（`SSL_ERROR_SYSCALL`），直连又被 reset。**Push/pull 用 SSH**：

```
git@github.com:<owner>/<repo>.git                  # 端口 22，可用
ssh://git@ssh.github.com:443/<owner>/<repo>.git    # 443，穿透性更好
```

**大传输容易被掐断**（`Connection to ... closed by remote host`）。超过几十 MB 的推送
建议拆成多个小提交、逐个推，并加保活：

```
$env:GIT_SSH_COMMAND = "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes"
```

`gh` 走的是另一条通道，API 一直是通的 —— 建仓、改可见性、查体积都用它。

### 测试

改 `afd-plugin` 或 `vllm-ascend` 时，**两个仓各自的 `AGENTS.md` 都强制要求带测试**。
新增环境变量必须登记在 `vllm_ascend/envs.py` 的 `env_variables` 字典里。

### 补丁

改动如果落到 `vllm-ascend`，按 `afd-plugin/AGENTS.md` 的规范走
`afd-plugin/compat/patches/npu/`，用 `# ### PATCH START/END` 标记差异。

---

## 七、设计文档

### 建议阅读顺序

```
先读   AFD代码分析对话整理.md                         AFD 怎么运作（架构入门）
再读   AFD_A2E_MXFP8量化_算子对接说明.md               算子怎么对接（可直接执行）
备查   AFD_A2E_E2A_量化链路源码调研与MXFP8方案设计.md   源码底账 + 对归档文档的勘误
归档   archive/AFD_Connector_MXFP8_方案.md            最早那版，核心结论错
```

### 清单

| 文件 | 用途 | 状态 |
|---|---|---|
| `AFD代码分析对话整理.md` | **架构入门**。CAMP2P 双通道（Gloo 控制面 + HCCL 数据面）、rank 拓扑与 token 数映射、A2E/E2A 角色判定、send op 为何有返回值、`NPUModelRunner.execute_model` 流程、Worker vs ModelRunner 分工 | ✅ 有效 |
| `AFD_A2E_MXFP8量化_算子对接说明.md` | **算子对接的唯一依据**。MC2 dispatch 的 OpDef 组合、tiling 校验、scale shape 规则、实现级搬运证据、a2e 要改的四层、框架侧改一行 | ✅ **最新**（关键声明我已用 `ops-transformer` 源码核对过） |
| `AFD_A2E_E2A_量化链路源码调研与MXFP8方案设计.md` | 源码调研底账 + **§九 的 E1~E10 勘误** | ⚠️ **§1~§8 有效（a2e/e2a 逐行解剖）；"方案设计"部分过时** —— 写在 0.30 升级、`W4A8MXFP` 确认、CANN 官方算子发现**之前**。**文件顶部有详细说明，引用前先读那段** |
| `archive/AFD_Connector_MXFP8_方案.md` | 最早一版方案（2026-09-23） | ❌ **已归档，核心结论不成立** —— "复用 INT8 通信槽零改码"是错的，照做会**静默算错**。仅 **§1.4 四路对比表**和 **§3.2 的 C5/C6 用例**仍有价值 |

**`archive/` 里的文件不要引用其结论** —— 只参考上表点名的那几节，其余以最新文档为准。

---

## 八、别做的事

- **别改 `vllm/` 和 `ops-transformer/`** —— 前者是上游、后者是第三方。它们的价值是"可核对的真相"，
  改了就没这个价值了。
- **别把 `_afd_env_repo/` 当源码改** —— 它是快照暂存，用脚本重建。
- **别在没确认 DP 配置的情况下改 MoE 通信路径** —— 走哪条路由 `world_size_across_dp` 决定，
  这个值是动态的。
- **别信"接口里有这个 dtype 组合"就等于"实现支持"** —— 在这套栈上已经被证伪过一次
  （`MoeInitRoutingV3` 的 OpDef 与文档不符，EP8 启动时崩）。要么跑最小实验，要么去 `op_tiling/` 看校验。
