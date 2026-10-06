# E1 — step v2 三次 kernel 收成一次 host 发射（出口判定：停）

日期：2026-10-06 ・ 前置：M4 结论「e2e 反超的前提是三次发射收成一次」的实证检验

## 融合入口结构

`leapquant-npu/gdn_lq_step_v2.py:step_pools_v2_fused`：一次调用完成
capture 检查 ×1、stream 获取 ×1、binding 解析 ×1，然后依次入队
dequant → gemv → epilogue（三个 op 模块各新增 `launch()` 快入口，
复用现有 executor 缓存键，miss 回退完整 wrapper）。三个 device kernel
不动（MIX 不可用、vector 写不了 L1、epilogue 要等 GM 里的 C——硬件
底线）。`step_pools_v2` 委托给融合入口；gdn.py v2 分支（两个调用点）
改传 raw state-index 视图；失败回退 triton step 的逻辑保留；
`VLLM_ASCEND_GDN_LQ_STEP_V2` 仍默认 0。

## micro wall（池化输入，生产形态，reset 计时外）

| B | host（入队 wall） | kernel（device） |
|---|-------------------|------------------|
| 16 | 0.24 ms | 0.41 ms |
| 64 | 0.23 ms | 0.43 ms |
| 256 | **0.21 ms** | **2.00 ms**（参照 2.22 ms 生产 wall） |

host 侧融合本身成立：0.21ms vs M4 账面的 ~6.4ms/层残差。

## e2e 对照（flag 开，SMOKE_MEM_UTIL=0.85，3 rounds 稳定）

| B | mode 3 现状 | 融合 v2（正确数值） | 判定 |
|---|-------------|---------------------|------|
| 16 | 96.9 ms/step | 184.9 | 差 |
| 64 | 136.7 | 205.0 | 差 |
| 256 | 278.9（出口线 ≤279） | **388.9** | **未过门** |

另有 IDXPOOL=1（idx 进池）变体：B=256 = 132.9 ms/step（1.69× mode 3、
~0.96× dense 139ms）——但数值损坏（见下），不作数。

## idx 进池的确定性损坏（root cause 未定位，证据链完整）

- 现象：idx 传入固定池槽（值/dtype/shape 与 fresh `.long()` 完全一致，
  逐 call 打印验证 match=True），真实引擎 48-token 冒烟确定性分歧
  @token 2（正确路径 @36）；**确定性**，非 flaky。
- 排除：idx 值错误（watchdog 在 kernel 执行后读槽，值完好）；
  竞态（idxdbg 全程 sync 仍 @2；且症状跨进程逐字节一致）；
  内存覆写（槽地址与 staging/ws/state 两两无交叠，pairwise 检查）；
  孤立环境全部 bit-exact——单/多 pools、每层不同 slot id、24 层
  复用同一 slot、异步交错设备工作，全部 0.000000。
- 两个 A/B 开关保留复现路径：`VLLM_ASCEND_GDN_LQ_IDXPOOL=1`（进池，
  快但错）、`VLLM_ASCEND_GDN_LQ_IDXWATCH=1`（执行后验槽）。
- 结论：e2e 残差的根子是 executor 缓存键随 fresh idx 的 data_ptr
  抖动（flush 的缓存持有首个 .long() 致后续地址漂移 → step 每步 miss
  慢路径，含 host sync）。进池是唯一快的排法，但撞上等价性破坏。
  该现象不抠了（停的条件已触发）。

## 出口判定：停

B=256 e2e 388.9 ≥ 279 出口线 → **step v2 留在 microbench，生产继续
triton step，flag 维持默认 0**。按纪律不再抠 cube 指令 / 不再调
kernel。融合入口与接线保留在库里（正确、更快、门控），供将来
executor 键问题（如 op 侧吸纳 idx 池化或 runtime 修复）解决后复测。

## 回归（终态，修正后默认路径）

48-token 冒烟 flag 开 3/4 @token 36 ✓；flag 关 3/4 @36 ✓；
五份基线全 PASS；test_step_v2 PASS（B=256 3-op wall 2.019 ms）。

## Commit

vllm-ascend `a3a46d3` feat(gdn): wire the fused step-v2 entry
(E1: single host launch, gate FAILED) —— gdn.py 接线（融合入口本体在
leapquant-npu，非 git 仓）。未 push。`gdn_leapquant.py` 的 M5 在途
改动（reconstruct_dense + quantize_from_tensor）保留在工作区未提交。
