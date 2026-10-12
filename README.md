# AFD A5 项目仓

Ascend A5 上 Attention/FFN 分离部署（AFD）的**唯一真源**：配套源码快照 + 版本清单 +
项目 patch + 执行脚本 + 排错规范 + 踩坑记录 + 当前进展。

蓝绿协作里它是传输通道：**蓝区**改完 push 这里的精确提交，**绿区** fetch 后按脚本运行。
绿区只读（只能 fetch / 改本地文件 / 跑进程 / 写 `/tmp`），**任何 push / upload 都是合规违规**。

## 版本清单

| 仓 | commit | 版本 | 上游 |
|---|---|---|---|
| `afd-plugin` | `bf5bd28` | — | https://github.com/Pingzii/afd-plugin |
| `vllm` | `ced6857afa` | `v0.30.0` | https://github.com/Pingzii/vllm |
| `vllm-ascend` | `8d4409d62` | `v0.19.1rc1-2448-g8d4409d62` | https://github.com/Pingzii/vllm-ascend |
| `ops-transformer`（CANN） | `951365ce8` | `v8.5.0-beta.1-6426-g951365ce8` | https://gitcode.com/cann/ops-transformer |

**这四个 commit 是一组经过验证的配套。**

## 注意

- **快照子目录（`vllm-ascend/`、`vllm/` 等）里不含 `.git`。** 所以那些目录不能
  `git diff` / `git log`；需要历史时回到上游仓 checkout 上表对应的 commit。
  **本仓自身的 `.git` 是完整的**，记录版本清单、patch、脚本、文档的每一次变更。
- **别手工改快照子目录** —— 它们是 pin commit 的原始快照。改动一律走 `patches/`。
- 各仓的 LICENSE 文件原样保留。它们的授权各自适用，本仓只是打包。
- 快照体积较大（约 420 MB），clone 时请留够空间。

## 目录

```
.
├── README.md                本文件
├── AGENTS.md                项目规范（与工作区根目录同一份，改一处要同步另一处）
├── record/
│   └── progress.md          ★ 当前进展 + 未闭环项 + 踩坑记录
├── docs/
│   ├── 排错方法论与技巧.md    方法论 10 条 + 技巧 5 条
│   └── 排错复盘_2026-10-12.md 8 个问题的现象 / 排查 / 对错 / 归因
├── patches/                 ★ 外部仓的改动（叠加在下面的快照上）
│   └── vllm-ascend/
│       └── dsv4-a5-cache-dtype/
│           ├── 0001-indexer-kv-dtype.patch   上游 PR #17635
│           ├── 0002-cache-dtype.patch        自己写的
│           ├── combined.patch                两个的拼接
│           └── README.md                      为什么 / 怎么用 / 怎么撤
├── scripts/                 ★ 自动化
│   ├── patch.ps1            蓝区：new / combine / check / apply / revert / list
│   └── verify.sh            绿区：check / apply / revert / status / baseline / compare
├── serve_scripts/
│   ├── serve_A.sh          # Attention 角色（2A2F）
│   └── serve_F.sh          # FFN 角色（2A2F）
├── afd-plugin/
├── vllm/
├── vllm-ascend/
└── ops-transformer/        # CANN 算子库
```

**下面这五个目录是源码快照，保持 pin commit、不带任何改动 —— 不要手工改。**
改动一律走 `patches/`，这样"这个 checkout 上打了什么"永远可查。

```
serve_scripts/  afd-plugin/  vllm/  vllm-ascend/  ops-transformer/
```

---

## 脚本

### 蓝区：`scripts/patch.ps1`

```powershell
# 从源仓的脏工作树提取 patch + 生成 README 骨架（然后手工把 README 填完）
.\scripts\patch.ps1 new     -Repo vllm-ascend -Name my-fix -Files vllm_ascend/platform.py

# 由 NNNN-*.patch 重新拼出 combined.patch
.\scripts\patch.ps1 combine -Repo vllm-ascend -Name my-fix

# 在干净树上确认整栈能打上（git apply --check）
.\scripts\patch.ps1 check   -Repo vllm-ascend -Name my-fix

# 应用 / 撤销（撤销会删掉 patch 新建的文件，并断言工作树已干净）
.\scripts\patch.ps1 apply   -Repo vllm-ascend -Name my-fix
.\scripts\patch.ps1 revert  -Repo vllm-ascend -Name my-fix

# 列出所有 patch 栈，并标出缺 README 的
.\scripts\patch.ps1 list
```

> 源仓默认是工作区下同名的目录（`-Repo vllm-ascend` → `../vllm-ascend`）。
> **脚本是纯 ASCII** —— PS 5.1 在没有 BOM 时按 ANSI 读 `.ps1`，中文注释会让整个脚本连解析都过不去。

### 绿区：`scripts/verify.sh`

三层验证，**第 3 层是最容易被跳过、也是唯一能抓住静默错误的那层**：

```
第 1 层  能打上        git apply --check              verify.sh check
第 2 层  服务能起      重启后 /v1/models 有响应        （--restart 可代跑）
第 3 层  输出等于基线  temperature=0 逐字节比          verify.sh compare
```

```bash
# 先建基线（在"已知正确"的状态下抓一次）
bash scripts/verify.sh baseline -o /tmp/ref.txt --prompt "Who are you?" --tokens 10

# 改了东西之后
bash scripts/verify.sh check  -r vllm-ascend -n dsv4-a5-cache-dtype
bash scripts/verify.sh apply  -r vllm-ascend -n dsv4-a5-cache-dtype
bash scripts/verify.sh compare -b /tmp/ref.txt --restart "bash serve_scripts/restart.sh"
bash scripts/verify.sh revert -r vllm-ascend -n dsv4-a5-cache-dtype

bash scripts/verify.sh status -r vllm-ascend
```

**只比 `temperature=0` 下的 `choices[0].text`** —— 整段 JSON 里有 `id` / `created`，每次都不同。

---

## 改动怎么流转

```
项目自有源码（afd-plugin 等）
    蓝区改 → 验证 → push 到本仓的精确提交
    绿区 fetch 该提交 → 按 scripts/ 运行

外部仓改动（vllm-ascend、vllm）
    蓝区按固定 base 生成 patch → 放进 patches/<repo>/<name>/ → push
    绿区 fetch → verify.sh check（apply --check）→ apply → 跑 → revert
```

**硬约束**

```
· 动手前先记基线（git status --porcelain），事后只撤本次变更，再核对工作树为空
· 不留没有撤销路径的改动
· 一次实验只变一个变量，并说明相对上次变了什么
· 结果只回最短结论，必要时附首错的少量上下文
· git apply --check 失败就停下，不许就地改目标文件
· 不把「未验证」写成「已完成」
```

---

## 验证状态约定

讨论任何结论时只用这三个词：

```
✅ 已验证            有可复现的证据
❓ 未验证            做了，但没验
⚠️ 已验证但前提已变   验过，但之后又改了别的
```

**当前各项的状态见 `record/progress.md`。**

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
