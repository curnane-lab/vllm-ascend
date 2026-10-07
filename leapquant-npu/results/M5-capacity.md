# M5 — 容量兑现：原生紧凑 composite checkpoint 池

日期：2026-10-07 ・ 模型：Qwen3.5-9B（`/mnt/workspace/models/Qwen3.5-9B`）・ 设备：Ascend 910B（卡 5）
协议：eager、greedy、`SMOKE_MEM_UTIL=0.75`（容量 ramp 用 `--mem-util 0.85`）、4K 前缀 + 48 decode tokens
数据文件：`runs/m5_cap.jsonl`（共享前缀 ramp）、`runs/m5_capu.jsonl`（unique 前缀 ramp）、`runs/m5_restore.jsonl`（restore 闭环）、`runs/smoke_t2.jsonl`（紧凑池 mode 3）

## 0. 改动一览（vllm-ascend，三个 commit，未 push）

| commit | 内容 |
|--------|------|
| `4f0a60b` | M5 在途基座：`reconstruct_dense` + `quantize_from_tensor`（gdn_leapquant.py +53 行） |
| `f2241c9` | **核心**：slot 2MB 稠密 → 1MB 复合（[H,8192] fp32 = 32×32KB head-block）；gdn.py 三分支 prefill 改 reconstruct/quantize 路径；Qwen3.5 model-class 形状 patch + `PlatformInterface._align_hybrid_block_size` 包装器（保活紧凑 mamba page 1,097,728B，不 pad 到 1,572,864B） |
| `ae3e6f5` | env 门必须 `int(os.getenv(...,"0"))`——`="0"` 是 truthy，曾炸 mode 0（MambaSpec assert） |

所有 M5 改动 env 门控（`VLLM_ASCEND_GDN_LEAPQUANT`），默认无 LeapQuant 时全走原路径；mode 3 默认行为不变。

## 1. 显存账

| 项 | mode 0（稠密基线） | mode 3（M5 紧凑池） | 比值 |
|----|--------------------|----------------------|------|
| GDN checkpoint slot | 2,097,152 B | 1,048,576 B（32×32KB head-block 复合） | **2.00×** |
| mamba page | 1,572,864 B（pad 到 attention page） | **1,097,728 B（保活紧凑）** | 1.43× |
| kv lcm block | 640 tokens | 384 tokens | 0.60× |
| 池容量（同 26.08 GiB KV 预算） | 575,595 tokens | 536,094 tokens（=1396 blocks） | 0.93× |
| vLLM "Maximum concurrency" | 70.26× @8192 tok/req | 130.88× @4096 tok/req | — |

要点：

- **slot 级 2.0× 兑现**——与 LeapQuant 论文的显存削减主张（41–56%）同向且更强（正好一半）。
- **系统池 token 数反而少 6.9%**：统一块把 mamba 字节绑到 attention token 粒度，384-token 块比 640-token 块每 token 多摊一份 mamba page（1,097,728/384 vs 1,572,864/640 = 2859 vs 2458 B/token）。块更小 × 每请求块数更多，把 slot 级的 2× 吃掉了大半。
- block 计数 1396（mode 3，384-token）vs ~899（mode 0，640-token，575,595/640）= **1.55× 块数**，调度/碎片角度更宽裕。

## 2. 正确性

### 2.1 紧凑池 vs 稠密池 decode 轨迹

`runs/smoke_t2.jsonl`（M5 紧凑池 mode 3）vs `runs/smoke_e3.jsonl`（M5 前稠密池 mode 3）：**4/4 prompt 全 48 token 逐 token IDENTICAL**。换池不改 decode 数值。

### 2.2 restore 闭环（prefix-cache 命中 → mamba state 重建）

`runs/m5_restore.jsonl`（hot 先发、冷发对照）：

| mode | chunked(512) | 结果 |
|------|--------------|------|
| 3 | 否 | 3/4 逐 token 精确；prompt0 共同前缀 45/48 |
| 3 | 是 | **与 unchunked 逐字节一致**（同样 3/4 + 45/48） |
| 0 | 否 | 2/4 精确；prompt1@36 发散、prompt2@0 全发散，**3 轮复现** |

- **mode 3 的 restore 比 mode 0 更准**：mode 3 hot-vs-cold 只差 prompt0 末 3 token（fp32 复合字节 reconstruct/quantize 的舍入），mode 0 的 align-mode memcpy restore 在本栈**确定性不准**。
- mode 0 缺陷是 **pre-existing**：mode 0 下 pools=None、块尺寸 640 未动、copy 机制未动，三轮复现同一发散点，与本轮改动无关。

### 2.3 出门冒烟（mode 3 flag 关 = 原路径）

cache 开/关均 3/4 @token 36 与基线一致（prompt3@36 为已知 pre-existing 分歧，M4 及以前就在）。

## 3. 容量 ramp

### 3.1 共享前缀（prefix cache 命中，4K + 48 decode，tok/s）

| N | mode 0 | mode 3 | 3/0 |
|------|--------|--------|-----|
| 48 | 306.1 | 225.6 | 0.74 |
| 64 | 391.1 | 249.3 | 0.64 |
| 80 | 461.0–475.7 | 278.7–284.2 | 0.60 |
| 96 | 505.5–533.8 | 282.8–285.0 | 0.55 |
| 112 | 571.6 | 304.0 | 0.53 |
| 116 | 581.4 | 286.4 | 0.49 |
| 120 | 576.8 | 291.2 | 0.50 |
| 128 | 614.3 | 299.5 | 0.49 |
| 144 | 634.4 | 292.4 | 0.46 |

- 两模式均**无线性 cliff / 超时失败**到 N=144（scheduler 上限附近）；mode 3 在 N≥80 饱和 ~285 tok/s，mode 0 持续爬到 634。
- TPOT 指标因 vLLM 0.30 metrics 属性名不符全为 null；以 wall/decode 做代理（N=96：mode 0 ~179 ms/tok，mode 3 ~276 ms/tok）。

### 3.2 unique 前缀（cache 失效，每请求自持块，memory-bound 版，tok/s，取第 2–3 轮）

| N | mode 0 | mode 3 | 3/0 |
|------|--------|--------|-----|
| 96 | 536.6–536.9 | 347.7–348.1 | 0.65 |
| 108 | 563.4–563.7 | 328.8–333.2 | 0.59 |
| 116 | 583.0–584.8 | 340.2 | 0.58 |
| 124 | 597.9–598.2 | 348.6–349.0 | 0.58 |

- **到 N=124 两模式都无内存 cliff**：解析边界（mode 0 ~116、mode 3 ~127，按池块数/每请求块数）未在实测中 binding——`llm.generate` 波次完成释放块 + 0.85 util 池更大，4K×48-decode 这一档内存不是先触底的资源。
- unique 前缀把 mode 3 的 decode 开销暴露得更干净：~0.58× mode 0（M4 已测 mode 3 单 batch 0.81×，高并发下 GDN 三分支 prefill 成本占比上升）。

### 3.3 判定

1. **slot 级 2.0× 显存兑现**：紧凑 composite checkpoint 本身达成设计目标。
2. **系统容量 1.4×（论文口径）未兑现**：实测 unique-prefix 无可测容量差（两模式都未到池边界）；按块数解析 ~1.09–1.55× 的理论优势被「块变小 × 每请求块数增多 × mamba 每 token 摊销变差」抵消，且 4K 短 decode 档先撞的是 scheduler/prefill 计算，不是池。
3. **真实收益点在共享前缀长尾 + 池余量**：mamba page 1.43× + slot 2× 让同样字节预算下 mode 3 的池在长上下文（8K+）/ 高并发长尾场景有更大余量——8K 档本轮只做了块数解析（1396/21 vs 899/13 ≈ 1.0×，mamba 每 token 摊销在 8K 与 640-token 块对齐时反而追平），未实测，记为开放项。

## 4. 副发现

- **mode 0 prefix-cache 命中 restore 确定性不准**（pre-existing，3 轮复现）：prompt1@36、prompt2@0。不是本轮引入；mode 3 的 restore 路径反而更准。报告如实记录，修复不在本轮范围。
- **env 门 truthiness 陷阱**：`os.getenv("X","0")` 返回字符串，`"0"` truthy。所有新增 env 门必须 `int(...)`（`ae3e6f5` 已修）。

## 5. 终态回归（M5 全量，exit 0）

| 门 | 结果 |
|----|------|
| test_pools_parity | PASS |
| test_step_triton | PASS |
| test_step_ascendc | PASS |
| test_flush_ascendc | PASS |
| test_flush_strided | PASS |
| mode 3 冒烟 flag 关 + cache 开（vs baseline_smoke4） | 3/4 IDENTICAL（prompt3@36 已知 pre-existing） |
| mode 3 冒烟 flag 关 + cache 关 | 3/4 IDENTICAL（同上） |
| test_flush_v2 --correctness-only | 5/5 PASS |
| test_due_gate | E5 MATRIX ALL PASS |
| test_m5_base（M5 在途基座） | ALL PASS |

## 6. 残留风险

- 不等页（attn 1.5MB vs mamba 1.05MB）只在 eager + align mode 验证；FULL/Piecewise 图模式 + 不等页未测。
- `_DUMP_STATE` 调试路径在 mode 3 现在读复合字节为 fp32（env 门控 debug-only，已知可接受）。
- 8K 前缀容量只做了块数解析，未实测（4K 网格已占满本轮单卡时间）。
- TPOT 指标需修 vLLM 0.30 metrics 属性名才能出数，本轮用 wall/decode 代理。
