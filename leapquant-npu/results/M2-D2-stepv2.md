# M2-D2：step v2 三段式实现结果（dequant → cube GEMV → epilogue）

日期：2026-10-06 ｜ 机器：Ascend910（soc ascend910_93，CANN 9.1.0，驱动 25.5.5）｜ 卡 5
提交（vllm-ascend，未 push）：`cc975a8` stage1 → `b2b6f7b` stage2 → `c2a6d34` stage3+stage2翻转 → `677532c` 集成

## 端到端计时表（2026-10-06 口径修正，见 results/M3-flush.md A 节）

**修正说明（grok 复核采纳）**：本表原始口径把 reset 放在计时循环内（与冻结的 M1 纯 kernel 口径不同）。修正口径 = 纯 kernel、reset 在计时循环外（与 dequant/gemv bench 完全一致）：

| B | dequant | gemv | epilogue（实测） | **3-op 纯 kernel 合计** | triton step（同口径重测） | 合计 vs 4.238（M1 冻结线） | vs 5.8（稠密 fla） |
|---|---|---|---|---|---|---|---|
| 16 | 0.054 | 0.046 | 0.075 | **0.176** | 0.270 | 0.04× | 0.03× |
| 64 | 0.152 | 0.080 | 0.227 | **0.458** | 1.096 | 0.11× | 0.08× |
| 256 | 0.555 | 0.414 | 0.855 | **1.824** | **4.062（复现 4.238，差 4%）** | **0.43×** | 0.31× |

**分口径结论：纯 kernel 口径下 v2 三 op 合计 B=256 为 triton 的 0.43×（赢 2.3×），全档胜出**；旧表"vs 6.031"作废——6.031 是循环内 reset 口径（reset 在 B=16 时占 ~1.8ms）。D1 报告"验收全出门"的对照线选择有误，以此表为准。

wall 口径（输入池化、executor 命中）：0.95–1.03 / 0.95–1.07 / 4.86 ms。**生产形态 wall（每步新 kn/qn 走慢路径）：5.75 / 9.67 / 29.3 ms —— 输入池化是生产化前置，详见 M3-flush.md B 节**。

三 op 纯 kernel 拆分（B=64/256 附近，单 op 计时，G=B×32）：

| op | B=16 | B=64 | B=256 |
|---|---|---|---|
| gdn_lq_dequant（AIV） | 0.045 | 0.139 | 0.549 |
| gdn_lq_gemv（AIC） | 0.035 | 0.063 | 0.420 |
| gdn_lq_epilogue（AIV） | 0.075（实测） | 0.227（实测） | 0.855（实测） |
| 三 op kernel 合计 | ~0.17 | ~0.45 | ~2.0 |
| 三 op wall（含 binding/reset/sync） | 0.98 | 1.62 | 4.97 |

*epilogue 为 M3-A 节实测值（reset 计时循环外）。B=256 的 cached-wall（4.86）与纯 kernel 合计（1.82）之差 ~3ms 是 binding Python 开销（3 次调用 ~0.75ms）+ host-device 同步间隙，不是单 op 问题；"每步新 kn/qn 走 executor 慢路径（含 host sync）"是生产形态 wall（29.3ms）的主因——bench 内是命中路径，慢路径只出现在生产形态（M3-flush.md B 节实测）。后续优化方向：输入张量池化 + binding 快路径瘦身。

## 对拍

**Stage 1（dequant）**：staging = fp16(codes) **逐位一致**（int8→fp16 精确）；aq 行 0/1 = fp16(kn·s_k/127)、fp16(qn·scale·s_k/127)，allclose 通过（约 1e-5 元素有 ≤1 ulp 差，来自 fp32 常量折叠顺序，数值无关）。G=512/2048/8192 全过。

**Stage 2（batched GEMV）**：cq 行 0/1（codesKn/codesQn）对 fp32 参考 **rel L2 = 0.000000**（fp16×fp16 积精确 + fp32 累加，与 CPU fp32 参考打印精度内一致）。G=512/2048/8192 全过。

**Stage 3（epilogue，step 全契约）**：vs `GDNLeapQuantPools.decode_step`，96 步 6 窗：
- o rel：[1.55e-4（首窗）… 9.2e-3（96 步）]，与 v1 已验证 kernel 的曲线**逐点重合**（v1 同种子 88 步处 9.49e-3 vs v2 9.63e-3）
- w/p：**bitexact**（torch.equal）；fill bitexact；kbuf 活行 bitexact
- u rel L2：窗内 ≤3e-3（任务线）✓；post-flush 3.98e-3（step 72），**v1 同点实测 3.94e-3**——是 v1 契约固有漂移（bf16 u_new 量化 + 两路重量化分歧），非 v2 缺陷，故 post-flush 容忍度按 v1 基线校准为 6e-3
- 窗末 decode_state rel：max 1.04e-2（v1：1.02e-2）

## 头几何核实（代码实证）

Qwen3.5-9B `config.json` text_config：`linear_num_key_heads=16`、`linear_num_value_heads=32`、head dims 128/128（`vllm_ascend/ops/gdn.py` 中 `rep = num_v_heads // num_k`）。**状态按 value head 持有**（pools H=32，ssm_state [slots,32,128,128]），value head h 用 key head h//2。**N=2 打包无意义**：两个 value head 的 codes（B 侧）不共享，N=2 只是 batch 打包没有复用。故 G=B×32 个独立实例，每实例一次 matmul 把 k'/q' 两行（M=16 pad）一起算。

## 架构形态（与 D1 结论一致）

MIX kernel 在本机不可用（fftsplus 空 kernel 也炸，D1 已证），三段式独立 op 靠 stream 序同步。scale 因式分解到 q/k 侧（staging = codes 纯 fp16 cast），cube 输入零数值损失；epilogue 从 v1 的 256 次/头 WholeReduceSum（codes 点积）降为 0，只剩 ~40 次合同/窗口点积。

## 集成与验收

- env：`VLLM_ASCEND_GDN_LQ_STEP_V2`（默认 0，集中定义在 `vllm_ascend/envs.py`）；flag 开且 LEAPQUANT=2/3 时走三 op（kn/qn 不 expand，GQA 展开收进 kernel）；任一 op 返回 False 自动回退 triton step；**mode 3 生产路径 flag 关时零改动**（gdn.py diff 仅在两个 decode 分支内新增 v2 子分支，repeat_interleave 收进 else）。
- 五份基线全绿：test_pools_parity / test_step_triton / test_step_ascendc / test_flush_ascendc / test_flush_strided。
- 48-token 冒烟（Qwen3.5-9B，4 prompts，temp=0）：
  - **flag 关**：与 `runs/baseline_smoke4.jsonl` 3/4 逐 token 一致（prompt 3 在 token 36 分歧——与现网行为一致，属既有近 tie）
  - **flag 开**：3/4 一致，**prompt 3 在完全相同的 token 36 分歧**（v2 轨迹与生产 flag-off 逐 token 相同）→ v2 无引入性数值差异

## 遇到的问题（大坑小坑全录）

1. **executor 缓存泄漏 2.6GB/步**（最大坑，engine OOM 根因）：binding 的 `_c()` 对 strided 的 ssm_state view 调 `.contiguous()` 实体化整个统一池视图，副本又被 `_CACHE` 永久持有。修复：传 `torch.as_strided(state,(1,),(1,))` 一元素基址视图（kernel 本就只按基址+stride 寻址）。v1 的 binding 有同样隐患（state 同样过 `_c`），**建议后续同样修复**（本次未动 v1 已验证代码）。
2. **IndexCheck 玄学**："custom op 后设备端高级索引必炸"的表象，最终至少两成是 **harness 自己的索引越界**（如 `idx=arange(b)+1` 超 SLOTS）触发 device 端 `(idx<bound)` assert trap，错误归因到后续无关 kernel。其余 genuinely flaky 的失败（同代码时好时坏、跨进程积累）疑似设备异常状态残留，重试可过。**所有 harness 已改为：op 之后不做设备端索引（CPU 快照/重建）**，集成路径（gdn.py）本身不含高级索引。
3. **两处移植笔误**（stage3）：cq 行 DataCopyPad blockLen 单位是字节（写成 4·F_V/8=64，应为 512）导致读回 UB 垃圾；漏抄 v1 的 `Duplicate(wq[32], f+1)` 导致 fill 恒写 1、窗口从第 3 步崩。staged parity（逐 op 对拍）分别定位。
4. **post-flush ubuf 容忍度**：按 v1 实测基线（3.94e-3）校准，非拍脑袋放宽。
5. 冒烟 OOM 另有 SMOKE_MEM_UTIL 余量问题（flag-on 需 0.75；v2 缓冲使碎片敏感，泄漏修复后应好转，未复测 0.85）。
6. D1 已知坑全部规避（repeat=1 reduce + 每发重设 mask、无运行时索引局部数组、128 行 Cast、DataCopyPad 四参）。

## 残留风险

- flag 开时每个 decode 步的新 kn/qn 张量使 executor 键不命中 → 走慢路径（含 host sync）。**生产化需把输入张量池化或改 executor 键策略**；当前 v2 flag 为研究门控，默认关。
- epilogue 未做 double-buffer，B=256 时 wall-kernel 差 ~3ms 有优化空间。
- "设备状态残留型" vec error 偶发（同代码重试即过），已用重试策略兜底；根因未定位（疑似驱动/firmware 级，chipId 7），建议换环境复验。
- v1 binding 的 state-contiguous 实体化隐患（见上 1）。

## 代码清单

- vllm-ascend（git）：`csrc/attention/gdn_lq_dequant/`、`csrc/attention/gdn_lq_gemv/`、`csrc/attention/gdn_lq_epilogue/`（均注册进 build_aclnn.sh ascend910_93 列表）；`vllm_ascend/envs.py`、`vllm_ascend/ops/gdn.py`
- leapquant-npu（非 git）：`gdn_lq_dequant.py` / `gdn_lq_gemv.py` / `gdn_lq_epilogue.py`（ctypes binding）、`gdn_lq_step_v2.py`（pipeline）、`tests/bench_dequant.py` / `tests/bench_gemv.py` / `tests/test_step_v2.py`
- 复现：`LD_LIBRARY_PATH=/home/developer/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH ASCEND_VISIBLE_DEVICES=5 python tests/test_step_v2.py`

---

## 生产化前置：输入张量池化（2026-10-06）

### executor 键结论

三 op binding 的 executor 缓存键 = 全部输入/输出张量的 **data_ptr 元组** + state.stride + (T, h, scale)（`gdn_lq_{dequant,gemv,epilogue}.py` 的 `_CACHE`；v1 的 `gdn_lq_step_ascendc.py` 同构）。生产形态每步新建 kn/qn/v/g_log/beta/o → 键不命中 → 慢路径（含 host sync）×3。

### 池化设计

`gdn_lq_step_v2.step_pools_v2` 内部维护 (t,nk,h,dev) 键控的**预分配输入槽**（kn/qn/v/g_log/beta/o 各一），每步 `copy_` 进固定槽位（异步流内拷贝），data_ptr 恒定 → 每步全命中。**关键发现：aclnn 运行时 executor 池按 op 类型 + desc 键（不含 data_ptr）**，为同 op 建第二个 executor 会复用内部块、悄悄废掉第一个——4 槽 ring 在复用步（第 5 步）输出全错（o rel 1.2），ring=1 单槽 96+ 步逐位级一致。故 **RING=1**（串行 decode 步间流序本就强制，无性能损失）；副作用：返回的 o 是池视图，仅保证到下一次 step_pools_v2 调用前有效（gdn.py 立即消费进 core_attn_out ✓，需久持请 .cpu() 快照）。另将共享 `_WS_BUF` 地板设为 256MB（防任何重分配）。

### wall 前后对照（ms/步，每步全新数据内容，设备端生成）

| B | 池化前 prod wall | 池化后 prod wall | cached wall | 加速 |
|---|---|---|---|---|
| 16 | 5.75 | **0.764** | 0.272 | 7.5× |
| 64 | 9.67 | **0.697** | 0.450 | 13.9× |
| 256 | 29.30 | **2.216** | 1.947 | 13.2× |

池化后 prod-cached 差 0.25-0.49ms = 设备端输入生成（5 个 torch op，模拟 projection 输出，非 step 成本）+ copy_。

### copy_ 开销（5 次拷贝合计，纯测量）

B=16: 0.104ms；B=64: 0.095ms；B=256: 0.076ms —— 在 +0.2ms 预算内。

### 验收状态

- step 全契约对拍：STEP V2 PARITY PASS（96 步 6 窗）
- 五份基线全绿；48-token 冒烟 flag 关 4.5s / flag 开 9.3s，均 3/4 逐 token 一致（token 36 既有分歧不变）
- v2 生产形态 wall B=256 = 2.216ms vs triton 同口径 4.066ms —— **生产形态下 v2 也赢了**
- 提交：池化全部落在 leapquant-npu（非 git 仓），vllm-ascend 零改动、无需 commit

### 残留风险

1. RING=1 的 o 生命周期契约（下一步调用即失效）——gdn.py 现用法安全，后续调用方需知悉。
2. executor-per-op-per-shape 限制：同进程多 (T,h) 组合并存时同 op 多 executor 的相互影响未系统性验证（本机 smoke/parity 单形状通过；多形状同进程建议每形状一轮或加进程级验证）。
3. 设备 flaky 依旧在（全部测量重试兜底）。
