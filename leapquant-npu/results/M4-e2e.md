# M4 — e2e 延迟矩阵 + ACL graph 回归门

日期：2026-10-06 ・ 模型：Qwen3.5-9B（`/mnt/workspace/models/Qwen3.5-9B`）・ 设备：Ascend 910B（卡 5）
协议：greedy、128 output tokens（graph 门用 48）、`SMOKE_MEM_UTIL=0.85`（FULL 另注）、三轮独立实测取末轮（带显存峰值）
数据文件：`runs/m4_e2e.jsonl`（第 1 轮）、`runs/m4_peak2.jsonl`（第 3 轮，peak 有效）、`runs/m4_graph.jsonl`（graph 门）

## 1. e2e 延迟矩阵（eager，9 格）

| mode | step v2 | B | tok/s | ms/step | 显存峰值 | v2/m3 比值 |
|------|---------|------|-------|---------|----------|------------|
| 0 | 0 | 16 | 204.2 | 78.1 | 57.35 GiB | — |
| 3 | 0 | 16 | 165.3 | 96.9 | 57.38 GiB | — |
| 3 | 1 | 16 | 87.0 | 183.6 | 58.17 GiB | **0.53×** |
| 0 | 0 | 64 | 703.0 | 91.4 | 57.43 GiB | — |
| 3 | 0 | 64 | 466.8 | 136.7 | 57.46 GiB | — |
| 3 | 1 | 64 | 313.4 | 203.9 | 58.29 GiB | **0.67×** |
| 0 | 0 | 256 | 1838.5 | 139.1 | 57.70 GiB | — |
| 3 | 0 | 256 | 917.1 | 278.9 | 58.99 GiB | — |
| 3 | 1 | 256 | 659.2 | 388.3 | 59.75 GiB | **0.72×** |

- ms/step = wall_s / 128 tokens × 1000（decode 每 seq 每 token 一步）。
- mode 0 与既有基线（204.5 / 703.9 / 1900.0 tok/s）在噪声内一致 → **生产路径无回归，树未变**。
- 三轮实测结论一致：v2 在全部三档 batch 上输给 mode 3（0.53×/0.67×/0.72×），B 越大差距越小但从未反超。
- 显存：v2 比 v1 多 ≤ 1 GiB（输入池 + staging），不构成约束。

## 2. 结论：v2 e2e 未赢 mode 3 —— 拆账

kernel 级 wall 时钟上 v2 有明确优势（B=256 生产 step：2.22 ms vs v1 triton 4.07 ms），但没有传导到 e2e。B=256 差值：mode3 278.9 ms/step → v2 388.3 ms/step，**v2 反而 +109 ms/step**。按 24 个 GDN 层拆账（注意是 24 层，32 是头数）：

| 项 | 每层 | ×24 层 |
|----|------|--------|
| kernel 省（4.07 → 2.22 ms） | ~1.85 ms | **~44 ms 省** |
| host 多（3 次 aclnn 发射的 binding/executor/launch） | ~6.4 ms | **~153 ms 多** |
| 净 | | **+109 ms/step** |

- v2 step 路径 = 3 个 aclnn op（gdn_lq_dequant AIV / gdn_lq_gemv AIC / gdn_lq_epilogue AIV）；host 开销与 kernel 无关，继续抠 cube 指令 e2e 不动。
- 瓶颈不在 kernel 而在 per-op 发射路径。**e2e 反超的前提是三次发射收成一次（单 op 化），或图内化且 FULL graph 可用**——这是设计决策，记为待 grok 定夺的开放项。**E1（2026-10-06 后续）已实证检验「收成一次 host 发射」：融合入口 + micro wall host 0.21ms/kernel 2.0ms 成立，但 e2e B=256 正确数值下 388.9 ≥ 279 出口线未过——根子是 idx 的 executor 缓存键抖动，唯一快的排法（idx 进池）撞上未定位的确定性等价性破坏。判定：停，v2 留 microbench，见 `results/E1-stepv2-fused.md`。**

**建议：`VLLM_ASCEND_GDN_LQ_STEP_V2` 维持默认 0（关）。** v2 保留为 kernel 级基准实现（三 op 正确、PIECEWISE 下数值精确）。

### 池化一节补记（隐患）

`gdn.py` decode 路径的 `idx = indices[:n].flatten().long()` 每层新建张量，executor 缓存键含 `idx.data_ptr()`：当前 caching allocator 对该尺寸的稳定复用使 data_ptr 不变、未触发慢路径；一旦分配模式变化（batch 尺寸抖动、显存压力）键会 miss 而每步退回 slow path（含 host sync）。启用 v2 前应把 idx 纳入输入池（ring slot + 稳定 data_ptr），与 kn/qn/v 同待遇。

## 3. ACL graph 回归门（B=16，48 tokens）

正确性判据：token_ids 与 mode3 eager B=16 基线（`runs/m4_e2e.jsonl`）逐 token 完全一致。

### PIECEWISE —— 10/10 通过

| 配置 | 5 次吞吐 (tok/s) | 均值 | ms/step | 精确匹配 |
|------|------------------|------|---------|----------|
| mode3 | 146.3 / 145.1 / 144.9 / 145.3 / 145.9 | 145.5 | 110.4 | 5/5 全 seq 全 token |
| mode3+v2 | 94.0 / 94.0 / 93.7 / 94.3 / 94.0 | 94.0 | 170.8 | 5/5 全 seq 全 token |

- 两形态 PIECEWISE 捕获/回放均稳定，数值与 eager 逐 token 一致 → **v2 的池化输入在 ACL graph 下时序契约成立**（step 先于 flush，fill=L 语义在捕获中不变）。
- 注意 B=16 下 PIECEWISE 比 eager 还慢（mode3：110.4 vs 96.9 ms/step，0.88×；v2：170.8 vs 183.6，0.93×）——小 batch decode 上图回放收益为负，与既有观察一致，仅作回归门不推荐使用。

### FULL —— 可运行但维持既有"已知不可用"结论

| 配置 | tok/s | vs eager | 正确性 |
|------|-------|----------|--------|
| mode3 @0.75 | 38.8 | 0.24× | 15/16 seq 精确；seq0 token 41 起漂移（7 token） |
| mode3+v2 @0.75 | 38.5 | 0.23× | 16/16 精确 |

- `SMOKE_MEM_UTIL=0.85` 下 FULL **捕获期 OOM**（capture 时额外申请 4.10 GiB 失败，已分配 49.93 GiB / 空闲 1.40 GiB）；降到 0.75 才能跑。`--cudagraph full` 需要比 piecewise 更高的捕获余量。
- 吞吐 0.23–0.24× eager，与既有 ~0.20× 记录一致。根因（已知，M3-4 记录，本次不改）：`gdn_lq_flush_ascendc` 在 capture 时 executor 未缓存返回 False → torch flush 回退被录进图；v2 三 op 同样因 `torch.npu.is_current_stream_capturing()` 检查回退 triton。v1 的 seq0 漂移即图内回退路径数值差异所致。
- **澄清**：v2 的「16/16 精确」是捕获期回退到 triton 的结果，不是三段式在图里跑；FULL 下没有任何 LeapQuant 自定义 kernel 进图。
- 结论：FULL 对两种形态都不构成可启用路径，维持关闭；v2 在 FULL 下数值仍精确（回退到与 v1 相同的 triton 语义）。

## 4. due-step 摊销附注

all-due flush（fill=L 后每 16 步触发一次全量 flush）单次实测 43.77 ms（B=256）。注意这笔账的层数口径：摊销 = 43.77 / 16 ≈ 2.74 ms/step **每层**，× 24 个 GDN 层 ≈ **66 ms/step，已含在 mode 3 的 278.9 ms 里（约 1/4）**——不是可忽略项，只是 v1/v2 两形态共用同一 flush 路径、与 v2 开关决策无关。v2 若启用同样省不掉这笔（flush 不变）；只有把 all-due 本身降下来的 cube 化（M3-3b）才动它。

## 5. M4 终态回归（vllm-ascend 零改动复核）

| 门 | 结果 |
|----|------|
| test_pools_parity | PASS |
| test_step_triton | PASS |
| test_step_ascendc | PASS |
| test_flush_ascendc | PASS |
| test_flush_strided | PASS |
| 48-token 冒烟（flag 关，3/4 匹配、分歧 @token 36） | 符合既有基线契约（`baseline_smoke4`） |

## 6. 变更与 commit

- M4 测量本身对 `vllm-ascend` 零代码改动；唯一改动是测量工具 `leapquant-npu/scripts/bench_e2e.py`（加 `--cudagraph piecewise|full`、`--tag`、jsonl 输出、npu-smi 显存峰值解析修复），该目录非 git 仓。
- 本报告的账目修正（拆账/摊销/结论/FULL 澄清/idx 隐患）随修正轮落地，关联的 M3-3b kernel 修复以两个 commit 进 vllm-ascend：`08323c4`（fprep 折 p + flags 语义）、`b39d4d2`（ftail sumAbs 清零 + payload 屏障），详见 `results/M3-flush.md` 3b 修正章节。
