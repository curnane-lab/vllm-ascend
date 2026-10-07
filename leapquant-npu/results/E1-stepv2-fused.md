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


---

# E1 有限定位实验（2026-10-07，grok 处方）——根因找到并修复

## 三变体对照（48-token 冒烟 flag 开 + pooled idx，真实 engine）

| 变体 | 行为 | 冒烟 |
|------|------|------|
| A 每次重建 executor | 正确 | 3/4 @36 |
| B 缓存 + 保留 `_CallContext` | 损坏，与 C **逐字节相同** | 0/4 @3 |
| C 现状（缓存 + destroy ctx） | 损坏（对照） | 0/4 @3 |
| B2 = B + 保留 ctypes attrs | **正确** | 3/4 @36 |
| **D = destroy ctx + 保留 attrs（默认）** | **正确** | 3/4 @36 |

## 根因（证据链）

缓存 executor 的 baked launch 引用（不是拷贝）了 slow path 上的
**ctypes attr 标量对象**（dequant/epilogue 的 `num_heads/num_k_heads/
slot_stride_bytes/scale`）。wrapper 返回后这些临时对象被 GC，内存复用，
之后每次缓存命中启动都读到垃圾 attr——`slot_stride_bytes` 损坏 →
kernel 用错误的 slot 步长寻址 state → 确定性损坏（B/C 输出逐字节一致、
任何 idx 形式、孤立 harness 全 exact 的谜题全部解释）。

- A 对：每次调用 attrs 新鲜。
- B 错且与 C 逐字节相同：**排除 ctx/描述符生命周期**（B 保活了 ctx 与全部描述符）。
- B2/D 对：attrs 是唯一悬空引用。D（destroy ctx + 保 attrs）是生产形态。

旁证：flush v2 的 fprep 缓存路径有同样的潜在悬空（其 attrs 侥幸存活于
常量复用），已按同一修法保活（`gdn_lq_flush_v2.py:_create` 返回 attrs）。

## 变体 B 的排除记录

B（保活整个 ctx）在 e2e 规模**死锁**（B=16/64/256 全部挂起 >15 分钟，
SIGTERM 不可杀）——ctx 整体保活会钉住运行时资源，长程运行必现。D 无此问题。

## e2e 规模验证状态

D 的 B=256 e2e 两次尝试被**环境楔死**阻断：LC=B 死锁被 kill -9 后设备
残留 in-flight ACL 操作，此后 B=64/256 的 epilogue `get_ws` 一律
ret=161001（C 对照同样失败——非变体问题，已判别）。小形状（冒烟）正常。
**e2e 门（B=256 ≤279ms/step）留待设备恢复后补测**，作为下轮 grouped GEMM
之前的第一个动作。

## E1 结论更新（替换本文件上一版"停"的结论）

- 根因：**attrs 生命周期**（已修，变体 D 为默认；`VLLM_ASCEND_GDN_LQ_STEP_LIFECYCLE` 保留 A/B/C/D 复现开关）。
- 冒烟 flag 开（STEP_V2=1，IDXPOOL=1 或默认 fresh-idx）= **3/4 @36 ✓**。
- e2e 门：两次复测（含静置 90s 后重试）均被环境楔死阻断（exit=124，每步
  epilogue get_ws ret=161001；C 对照同样失败，非变体问题）。不宣布过、不宣布停；
  下轮（grouped GEMM 之前）第一个动作 = 设备恢复后补测 B=256 e2e（D 配置：
  STEP_V2=1 + IDXPOOL=1）。132.9ms 数字保持作废。


---

# E1 e2e 门补测尝试（2026-10-07 上午）——设备楔死未恢复，门保持未判定

## 设备状态核验

- 最小探针：小分配+matmul ✓；flush v2 @B=256/64 ✓（v2 flush 族不受影响）。
- **step v2 的 epilogue op 族楔死**：`get_ws ret=161001`，范围从 t≥64 扩散到
  warm 的 t=1；跨进程确定性复现；standalone bench（无 engine、无变体）同样失败。
- 恢复手段逐项排除：残留进程（无，npu-smi 显示零进程）；npu-smi 无
  非特权 reset 命令；无可清理的用户态 kernel 缓存（仅 atrace 日志）；
  HBM 3116/65536MB 无泄漏性耗尽；驱动级复位工具（device_boot_init.sh 等）
  需 root，不在授权范围。
- 判别：楔死非变体/软件配置问题（C 对照与 D 同样失败，且在楔死前同一
  配置可运行），系 LC=B 死锁被 kill -9（带 in-flight ACL 操作）后的设备/
  驱动侧持久状态；且每次 161001 失败后似乎在恶化（t=1 warm 由通过转为失败）。

## e2e 门状态

**未判定（BLOCKED，非失败）**：B=256 ≤279ms/step 的测量需要一个健康的
epilogue op；当前任何 t≥1 的 epilogue get_ws 都失败。恢复手段需管理员
（驱动/设备复位）。恢复后的第一个动作（已备好命令）：
`STEP_V2=1 IDXPOOL=1 bench_e2e B=16/64/256`。

## E1 最终结论（截至本轮）

- 根因：缓存 executor 引用 ctypes attr 临时对象——**已定位、已修复
  （变体 D 默认）、冒烟 flag 开 3/4 @36 已验证**（上轮）。
- e2e 门：未判定。按停止规则不宣布过、不宣布冻结——状态为
  **"修复待设备恢复后补测"**。132.9ms 数字保持作废。

## 勘误（2026-10-07，grok 代码审查）

**变体 D 的冒烟验证无效**：epilogue 的 `attrs`（h/nk/s_slot/scale）建出来了、缓
存也保活了，但调用是 `GetWorkspaceSize(*args, ...)`——attrs 从未传进 autogen
接口（签名里它们在 `beta` 与 `out` 之间）。慢路径必失败 → 返回 False → 融合入
口回退 triton step。triton 现网本身就是 3/4、分歧 @36，所以「D 正确」与「根本没
跑到 epilogue」不可区分。已修：`*attrs` 插入 args（beta 后、o 前，对齐 autogen
签名）；文件头注释改为如实描述（默认 D，及本勘误）。**补测前置条件**：确认无
`[gdn-lq-epilogue] slow-path exception` 日志、`VLLM_ASCEND_GDN_LQ_STEPlog=1`
下确认缓存命中；设备须先从 kill -9 楔死恢复（epilogue ret=161001）。过门线仍
是 B=256 ≤279ms/step。132.9ms 继续作废。
