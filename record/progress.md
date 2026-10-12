# 当前进展

> 最后更新：2026-10-12
> **验证状态只有三种**：✅ 已验证（有可复现证据） / ❓ 未验证（做了但没验） / ⚠️ 已验证但前提已变。
> 不写"应该没问题"。

---

## 一、结论速览

| 事项 | 状态 |
|---|---|
| A5 上 DeepSeek-V4 能起来、输出正常 | ✅ **已验证** —— 加两个启动参数即可 |
| `cache_dtype` 修复（patch 栈 `dsv4-a5-cache-dtype`） | ❓ **未验证** —— 写好了、两个 base 上 `git apply --check` 通过，但**没在 A5 上跑过** |
| vllm-ascend 的上游 PR | ❓ **未开** —— 分支已推送 |
| **A2E 通信量化（本项目目标）** | ❌ **未开始** —— 一行代码没写，门 1 也没做 |

---

## 二、进行中的事

### ① 让 DSV4 在 A5 上起来 —— ✅ 完成

**问题**：升级到 vLLM 0.30 + vllm-ascend `8d4409d62` 后起不来。

**根因（两个配置源各自错）**：

```
cache_config.cache_dtype          = "auto" → 解析成 bf16，而 A5 只有 FP8 实现
attention_config.indexer_kv_dtype = "auto" → 同样解析成 bf16，
                                     而 vLLM 文档规定 DeepSeek 稀疏 indexer 的 auto 是 fp8
```

**当前可用的两条路**（都验证过）：

```
路线 A：启动脚本加两个参数
    --kv-cache-dtype fp8
    --attention-config '{"indexer_kv_dtype": "fp8"}'
    → ✅ 已验证：服务起来、输出正常

路线 B：打 patch 栈 patches/vllm-ascend/dsv4-a5-cache-dtype/
    → ❓ 未验证：脚本还没在 A5 上跑过
```

**已验证的输出基线**（`temperature=0`，用于 `verify.sh compare`）：

```
prompt:  "Who are you?"   max_tokens: 10
text:    "\",\n  \"I am the one who is the"
```

### ② patch 栈待验证

```
patches/vllm-ascend/dsv4-a5-cache-dtype/
    0001-indexer-kv-dtype.patch     上游 PR #17635（2026-10-08 合入 main）
    0002-cache-dtype.patch          自己写的，上游至今未修
    combined.patch                  两个的拼接
    README.md                       为什么 / 怎么用 / 怎么撤
```

**注意**：`#17635` 能干净地打到 `8d4409d62`（我们的 pin 比它早 11 天），
所以 `0001` 用的是上游代码，不是自研 workaround。**升级到含 `#17635` 的版本后要丢掉 `0001`。**

**验收步骤**：

```bash
cd <项目私仓>
bash scripts/verify.sh check  -r vllm-ascend -n dsv4-a5-cache-dtype
bash scripts/verify.sh apply  -r vllm-ascend -n dsv4-a5-cache-dtype
#   然后从 serve 脚本删掉那两个参数，重启
bash scripts/verify.sh compare -b <基线文件>
bash scripts/verify.sh revert -r vllm-ascend -n dsv4-a5-cache-dtype
```

### ③ vllm-ascend 的上游 PR

```
分支  Pingzii/vllm-ascend:fix/dsv4-a5-cache-dtype
base  upstream main（已含 #17635，所以 PR 里只有 cache_dtype 那一半）
改动  vllm_ascend/platform.py +44、tests/ut/test_platform_dsv4_cache_dtype.py +63
```

**开 PR 之前必须先过 ②** —— commit message 里写了"A5 验证结果会回填到 PR 上"。

---

## 三、未闭环项

```
① patch 栈在 A5 上的验证（验收步骤见上）
② vllm-ascend 的 PR 未开
③ ★ 门 1：A2E 占单层耗时百分之多少 —— 零改码的 profile，一直没做
     判据：<3% 止损 / 5~15% 权衡 / >20% 值得做
     这是决定 A2E 量化（约 540 行）值不值得写的唯一前置
④ AFD_A2E_MXFP8量化_算子对接说明.md 全文用 H=7168，本模型是 H=4096
     它被 AGENTS.md 指定为"算子对接的唯一依据"，读者会算错字节数
⑤ 排错方法论里缺一条：「刚改过什么，就先怀疑什么」
```

---

## 四、踩坑记录

**只记代价换来的、下次还会再踩的。**

| 坑 | 代价 | 下次怎么避 |
|---|---|---|
| 看到 `.view` 崩就去改 `.view`，而根因是 cache 被解析成 bf16 | 一整轮 + 一个错补丁 | 报错点是"第一个发现不一致的地方"，不是出错的地方 |
| patch 改错了 `block_stride` 语义（×32），服务起来了但输出是词沙拉 | 一整轮 | 只验"不崩"不算过，要和基线逐字节比 |
| 加了 fp8 flag 之后，之前"cache 非连续"的前提其实已失效 | 同上 | 引入新变更后，基于旧状态的推理要重过一遍 |
| 差点给 afd-plugin 提一个上游 11 天前就做完的 PR | 差点一整轮 | 提上游前先查 issue/PR |
| 探针的派生值在 `.view(torch.uint8)` **之前**算的，差 2 倍 | 一轮往返 | 报告派生量要说明它是对哪个对象、在哪个时刻算的 |
| 提交信息里写了没做过的 A5 验证 | — | 验证状态必须分类；自己复查时抓到了 |
| `.patch` 被 CRLF 转换破坏过 | — | `.gitattributes` 里 `*.patch text eol=lf` |

---

## 五、怎么用这个仓

```
README.md              项目总览、版本清单、布局
AGENTS.md              项目规范（与工作区根目录同一份）
docs/                  排错规范、复盘、设计文档
patches/<repo>/<name>/ 外部仓的改动（叠加在快照上）
scripts/               自动化：patch.ps1（蓝区）、verify.sh（绿区）
record/progress.md     本文件
<四个源码快照> + serve_scripts/   配套源码（保持 pin commit，不带改动）
```

**快照目录（`vllm-ascend/`、`vllm/` 等）不要手工改** —— 它们是 pin commit 的原始快照，
改动一律走 `patches/`，这样"打了什么"永远可查。
