# E5 — 地基修复（oracle / due 镜像 / 可信测量）

日期：2026-10-06 ・ 触发：GPT/grok 代码审查（对 1b46e36，主机逻辑反例，未跑 NPU）推翻 flush v2 两块根基
范围：`leapquant-npu`（测试与驱动，非 git 仓）+ `vllm-ascend`（撤接线 revert + pools epoch）

## 0. 撤接线

`8637cc7`（E2 接线）已 revert（`3865c83`）：gdn.py 恢复纯 v1 flush，flush v2 只在测试路径可达。`c6af342`（fprep 写 slot id）保留——镜像修复后有用。生产 flush = v1 不变。

## 1. Oracle 修正（P1）

`tests/test_flush_v2.py:simulate_true` 原实现三处错（GPT 定性）：用未施加本步 decay 的 S 算更新；累计衰减重复乘；窗口项循环内加、返回又加。修正后递推（与 `pools.decode_step` 语义逐点一致——其重建 s 是全衰减态，`upd = β(v − s·k)`）：

```
S̄ = a · S_prev;  u = β · (v − S̄ k);  S = S̄ + u kᵀ
```

**主机侧标量反例回归**（无需 NPU，先跑）：S₀=1, a=0.5, k=v=β=1 是不动点 1.0——修正后 16 步恒 1（assert <1e-6）；未衰减更新递推漂移收敛到 2/3（0.666672，assert 验证；GPT 对原始双计账 oracle 的标量反例是 16 步 1.501064）。`ORACLE SCALAR PASS`。

**重测结论（正确 oracle 下）**：

| 判据 | 结果 |
|------|------|
| torch flush 对正确 oracle 绝对误差 | **0.00181**（旧 oracle 下 0.0524——旧参考自身错了 30×） |
| kernel 对正确 oracle 绝对误差 | **0.00183–0.00184**（门 0.005，**过**） |
| kernel/torch 误差比 | 1.016（informational；1.01 门是按错误 oracle 标定的，现被噪声淹没，已撤 assert） |
| 同一 pre-flush 状态克隆直接对拍（kernel vs torch） | rel = **0.00116**，w/p/fill 全等 |
| 非 due 槽不变性 | 字节级一致（int8 视图；fp32 `torch.equal` 对 -0.0/NaN 位模式有伪差，已修比较路径） |
| 多窗口（6 窗 96 步 forced） | 每窗 rel ≤ **0.00119** |

**结论：flush v2 正确性在正确 oracle 下真实成立**（绝对误差过门）。此前 claim 的 "ratio 1.0000" **撤回**——那是在错误 oracle 上的比值，数值上巧合没有意义。

## 2. due 镜像修复（P1）

GPT 反例：`_MIRROR` 以 `str(dev)` 为键 → 同设备同形状的不同层共享 fill/age（层 A 留 fill=1 镜像，层 B 实际 fill=16 被跳过，零发射返回 True）。E2 过程中还实测了共享镜像在生产异步 pacing 下间歇漏 fold（parity 2/3 失败、冒烟 @18 而非 @36）。

修法（全部落地为行为，非注释）：
- **键 = (device, id(pools))**：每层各持镜像；同形状两 pools 完全隔离（MATRIX two-layers / gpt-mock 验证）。
- **slot 生命周期 epoch**：`GDNLeapQuantPools.init_from_dense` 递增 `_lq_epoch`（生产 resync 入口）；镜像在 epoch 变化时清空全部 fill 记录 → 槽复用后一律重新保守学习（MATRIX slot-reuse-epoch）。
- **未知/失效一律保守发射**（打满 8 级）：view 缺失/过期、槽不在字典、epoch 失效，全部 launch（ftail 设备侧复查保底）。
- **fill≥L 语义修正**：记录的是 fold 当步的 pre-fold 读数。显式 step 计数下 fold 后窗口推演 = `cur − s0`（真实 fill 走 1,2,…，**16 步后再 due**）；同一步双 flush（cur==s0）重发射；**step=None 时一律重发射**（无 decode 的测量循环里连续 fill≥L 读数永不跳过——旧 all-due 计时被 host skip 污染的漏洞就此堵死）。
- 发射时 prune 不在批内槽的记录（离场槽的 sim 只增不减，不 prune 会成 phantom launch 源）。

**回归矩阵（tests/test_due_gate.py，逐项全过）**：T1 键=slot id+未见槽发射（1,5,18 节奏）；T2 稳态 16 步窗恰好一次全量（+冷启动）；T3 换位 hiding 失效（A 恰好 16 发射、B 19）；T4 nothing-due 纯 host（中位 0.086–0.108ms）；MATRIX：两层同形状独立镜像 / 槽复用 epoch 失效 / fill=0·15·16 转移+credit+重发射 / 混合 due 批 / 连续 96 步 6 窗（发射恰在 1,16,…,96）/ B=16·64·256 形状覆盖 / GPT mock 反例固定回归。

## 3. 可信测量（P2，旧数字作废）

旧 "all-due 26.9ms" 被镜像污染（连续 flush 无 decode、fill=16→base=0 的 host skip 混进平均），**撤回**。新脚本 `tests/bench_flush_v2_measure.py`，三组分开测，原始 jsonl 在 `runs/flush_v2_measure.jsonl`：

| B | ① forced all-due（绕镜像纯 8 级） | ② 真实 step→flush 4 窗 64 步（发射数=5，含 decode 的每步摊销） | ③ nothing-due 干净首态中位 host |
|---|------|------|------|
| 16 | 4.92 ms | 5.60 ms/step | 0.100 ms |
| 64 | 10.11 ms | 6.55 ms/step | 0.103 ms |
| 256 | **33.52 ms** | 15.57 ms/step | 0.156 ms |

- ②的每步摊销含 decode_step 设备时；纯 flush 摊销 = 发射数×①/64：B=256 ≈ 5×33.52/64 = **2.62 ms/step**，对照 v1 的 43.77/16 = 2.74 ms/step——**kernel 级 v2 仅 1.05×**，远非旧账面的悬殊差异。
- ③的 max 尖峰（2–5ms）来自每 4 步的 view 刷新拷贝的 doorbell 延迟，生产 pacing 下不影响（发射路径才 cares）。

`bench_e2e.py` 口径修正：ms/step 标注为「生成墙时摊销（含 prefill）」；warmup 改为满 batch 形状；npu-smi 峰值列改注「运行结束单采样，非瞬态峰值」（列名 pool_at_end_gib）。报告（M4 等）中 153ms host 等残差推算项全部视为**推算**。

## 4. 作废声明清单

1. **"对拍 ratio 1.0000" 撤回**：错误 oracle 上的巧合比值；正确口径 = 绝对误差 0.00183（门 0.005，过）+ 直接对拍 rel 0.00116。
2. **"all-due 26.9ms@B=256" 撤回**：镜像污染；forced 重测 33.5ms。
3. **E2 e2e 数字（279→262 等）撤回**：接线已 revert；且其中一轮实为 NameError 静默回退 v1 的假数据（修复前），不具参考价值。
4. M3-flush.md / M4-e2e.md 中基于上述数字的推论以本文件为准。

## 5. Commit 与残留

- vllm-ascend：`3865c83` revert 接线；` pools epoch` 单独 commit（只提 init_from_dense 的 epoch 行，M5 段落留在工作区）。`c6af342` fprep slot id 保留。未 push。
- leapquant-npu（非 git 仓）：oracle 修正、镜像 v3、回归矩阵、测量脚本均在工作区。
- 残留：① step v2 的 pooled-idx 确定性损坏仍未定位（E1 遗留，与本轮无关）；② 镜像 view 每 4 步刷新是准确性/开销的折中（≤4 步滞后窗 << 15 步 due 期，安全）；③ 发射合一 / grouped GEMM 未动（按处方在镜像修复后才可议）。
