# M3：flush 图内化 结果报告

> **勘误（E5，2026-10-06）**：本文件 M3-3b 章节的「对拍 ratio 1.0000」与「all-due 26.9ms」
> **撤回**——前者建立在错误的测试 oracle 上，后者被 due 镜像污染。修正后的口径
> （oracle 修正、镜像修复、三组可信测量、作废清单）以 `results/E5-foundations.md` 为准。

日期：2026-10-06 ｜ vllm-ascend 提交链：`a9d4110`(M3-1) → `b84e86e`(M3-2) → `d3b9950`(binding 修复)
（M3-3 尝试过、实测回退，未提交，见下）

## 子项完成状态

| 子项 | 状态 | 证据 |
|---|---|---|
| M3-1 fill=L 语义 | ✅ 完成（三重门全绿） | 5 对拍 PASS + v2 parity PASS + 48-token 冒烟 3/4（分歧仍在 token 36） |
| M3-2 flush due 扫描 | ✅ 完成 | nothing-due 0.046/0.078/0.051ms @B=16/64/256（≤0.1ms 达标）；all-due 不退化 |
| M3-3a flush S→fp16 | ❌ 实测回退 | 见下（更慢 + 判据余量变小，已 checkout） |
| M3-3b flush 三段式 cube | ✅ 全链贯通（修正轮修复 4 个代码 bug：staging undersize / 漏 p / sumAbs 未清零 / executor 池 phase 碰撞） | 对拍 ratio 1.0000，延迟表见 3b 章节 |
| v1 binding 修复 | ✅ 完成 | test_step_ascendc / flush / strided parity 不变（PASS） |
| FULL-graph 复测 | ⏸ 未复测 | 代码路径未变（capture 时 binding 返回 False → 图内录 torch flush），piecewise-only 限制维持 |

## A. 计时口径修正表（纯 kernel，reset 一律计时循环外，两轮区间）

| B | dequant | gemv | epilogue（实测） | 3-op 合计 | triton step（同口径） | 合计 vs 4.238 | 合计 vs 5.8 |
|---|---|---|---|---|---|---|---|
| 16 | 0.054 | 0.046 | 0.075 | **0.176** | 0.270 | 0.04× | 0.03× |
| 64 | 0.152 | 0.080 | 0.227 | **0.458** | 1.096 | 0.11× | 0.08× |
| 256 | 0.555 | 0.414 | 0.855 | **1.824** | **4.062** | **0.43×** | **0.31×** |

**triton 同口径（reset 圈外）重测 = 0.270 / 1.096 / 4.062 —— B=256 复现了冻结线 4.238**（差 4%，同口径确认）。D1 的 2.097/3.018/6.031 是"循环内 reset"口径：reset（th_index 散写）在 B=16 时占 ~1.8ms，完全淹没 step 本体。**修正结论：纯 kernel 口径下 v2 三 op 合计在 B=256 为 triton 的 0.43×（赢 2.3×），B=16/64 更是 0.65×/0.42×**；旧报告"对照 6.031"的口径作废。epilogue 实测 0.075/0.227/0.855（此前报告中的 ~1.0ms 是 wall 差值估的，实际纯 kernel 更便宜；wall-kernel 差是 binding/reset/同步开销，不是 kernel）。

## B. 生产形态 wall（每步新建 kn/qn → executor 慢路径，含 host sync）

| B | cached wall（输入池化） | prod wall（生产形态） | 慢路径可见开销 |
|---|---|---|---|
| 16 | 0.95–1.03 | **5.75** | ~4.8ms/步（3 次慢路径 executor 创建，各含 torch.npu.synchronize） |
| 64 | 0.95–1.07 | **9.67** | ~8.7ms/步 |
| 256 | 4.86 | **29.3** | ~24.5ms/步 |

**e2e 可用性结论：v2 flag 按生产形态（新输入张量每步）当前不可用**（B=256 墙 29.3ms ≫ triton 4.24ms）。输入张量池化（稳定 data_ptr → executor 常驻快路径）是生产化前置条件；池化后墙 ≈ cached wall + ~0.7ms binding 开销。flag 默认 0，此风险不触达生产。

## flush 延迟表（M3-2 现状 = 当前 HEAD）

| B | step（triton，纯 kernel） | all-due flush | nothing-due flush | due-step 尖峰（step+all-due） | 16 步摊销（step+nothing-due+all-due/16） |
|---|---|---|---|---|---|
| 16 | 0.270 | 2.78 | 0.046 | 3.05 | 0.49 |
| 64 | 1.096 | 11.10 | 0.078 | 12.2 | 1.87 |
| 256 | 4.062 | 43.77 | 0.051 | 47.8 | **6.85** |
| 256（v2 step） | 1.824 | 43.77 | 0.051 | 45.6 | **4.61** |

all-due 44ms@B=256 是 v1 vector 实现的算法固有成本（每头 ~2688 次标量 S-pipe 读 dominate，~90%）；M3-3a 证实单纯换 fp16 无助于此（反而更慢）。

## M3-3a 回退记录（实测依据）

改动：S 在 UB fp32→fp16（64KB→32KB），pass 边界插 cast。结果：**all-due 3.25/12.8/50.5ms（比 fp32 的 2.78/11.1/43.8 慢 ~15%）**——每行多一次 fp16→fp32 cast 的成本超过任何向量化收益，且 kernel-vs-torch checkpoint rel diff 末窗 0.0115（fp32 版 0.0102，本机 run 间方差内但余量变小）。**已 checkout 回退**，flush 保持 b84e86e 的 fp32 实现（五份基线复测全绿）。

## M3-3b 设计（未动工，后续工作）

all-due 的唯一下降通道是把 S 重建与三个投影搬上 cube（仿 step v2 三段式，stream 序三 op）：
1. AIV prep：codes 行 ×sk/127·sv·p + q^T·u 项 + ubuf^T·(w⊙kbuf) 项装配成 fp16 S staging [G,128,128]（dequant op 形态可复用；三个 rank 更新实际是 3 个 batched GEMM 的 A/B 装配）；
2. AIC batched GEMV（复用 gdn_lq_gemv 形态）：m1=S^T q0 [K,16]、qv=S·qk [V,16]、uk=S^T qv [K,16]（B 侧转置读，C 行连续）；
3. AIV 收尾：两次 Mgs4（各 12 个点积，~24 次 reduce/头，从 256 降到可忽略）+ E/sk/sv/codes 量化。
估算改动量：2 个新 op（prep AIV + GEMV AIC 扩展 N 语义）+ flush 主体重写，参照 step v2 的实际调试成本（~10 个构建-测试循环），目标 all-due ≤5ms@B=256。

## e2e 矩阵（48-token 冒烟，Qwen3.5-9B，4 prompts，temp=0，SMOKE_MEM_UTIL=0.75）

| 形态 | 耗时 | tok/s | 逐 token 对基线 |
|---|---|---|---|
| mode 3 现状（triton step + flush，M3 后） | 4.5s | 42.7 | 3/4 一致，分歧在 token 36（既有近 tie，全程未变） |
| mode 3 + STEP_V2=1（生产形态 wall） | 9.0s | 21.3 | 3/4 一致，同一 token 36（数值与生产逐 token 相同） |

B 档缩放的 e2e 需大并发压测环境（当前冒烟 4 prompts）；kernel 级 B 档数据见上表。fp32 稠密基线（生产 fla kernel ~5.8ms@B=256 层延迟）以冻结参考线列入，未在本机复测 fla kernel。

## 时序依赖（fill=L 后的语义契约，报告备案）

flush(idx) 不得对同一头连续调用两次而中间无该头的 step：gdn.py 与测试 harness 均为"每步先 step 后 flush"且 flush 只含当批 idx，模式不可达；w=0 使偶发二次 fold 的窗口贡献为零，但会把已折 checkpoint 再量化一次（退化不腐化）。M3-1 前该契约由"flush 清 fill"隐式保证，M3-1 后由调用顺序保证——**图内化（graph capture 把多步录进一张图）时若重排 step/flush 顺序必须保持此序**。

## 残留风险

1. 设备偶发 vec error（跨进程残留型，重试即过）仍在，所有测量均已重试兜底；根因未定位（疑似驱动级）。
2. v2 flag 生产形态 wall 高（B 表）——输入池化前不可开。
3. all-due flush 44ms@B=256 已用 v2 cube 链降到 27ms（仍未达个位数，方向 = grouped GEMM）；v2 未接入生产路径，见 3b 章节。
4. FULL-graph capture 未复测；piecewise-only 限制维持原结论。
5. leapquant-npu 侧 binding（gdn_lq_*.py）的 warm_cache/dummy 几何与 key 策略同 v1 隐患已修；leapquant-npu 非 git 仓，变更以文件为准。

---

# M3-3b 追加章节（flush 三段式 cube，2026-10-06 收尾）

## 设计落地

三个新 op（commit `a1af622`，**未接入 gdn.py**，v1 flush 仍是唯一 flush 路径）：
- `gdn_lq_fprep`（AIV）：due 扫描 + 装配 staging。核心技巧：**一次 GEMM 算出 S**——D 项（codes 反量化）不写死，而是 A 的行 32-159 = diag(s_v·p)（128 次标量写）、B 的行 32-159 = codes 行 ×(s_k/127)（连续拷贝+向量乘），K-dim=192。全程零转置（cube 的 isTranspose 做逻辑翻转）。
- `gdn_lq_fgemv`（AIC，phase 1-4）：S=A^T B（fp16 C）→ m1=S^T q0 → qv=S·qk → uk=S^T qv'。
- `gdn_lq_ftail`（AIV，phase 1-3）：两次 Mgs4（写预转置 [K,32] staging 喂 cube 直读）+ E/sk/sv/codes 量化写回。

## 趟出的平台事实（已写进代码注释）

1. **AIC 上 GlobalTensor::GetValue 标量读 GM 必然故障**（假"全实例处理"对照实验证实）→ fgemv 无法 kernel 内判 due，改 fprep 写 flags（fill 值）+ host 端无同步 due 门（pinned 镜像模拟，见 3b 修正章节；最初的 `flags.sum().item()` 实现已废弃）。
2. **转置 B 的 ND 读在 G>64 时 MTE 非法地址**（T=2 过、T=8 崩，两次复现）→ kq/vq 由 AIV 预转置成 [K,32]，cube 直读。
3. **mgs 结果的裸 TBuf→GM DataCopy 故障** → 改走 v1 验证过的 TQue payload 路径后 t1 通过。
4. MatmulApiTiling 拒绝 N=16（→32）、拒绝复用 SetShape 的对象（→双 tiling 对象）。

## 卡点（2026-10-06 修正轮：已全部解决，全链贯通）

原记录「ftail phase-2 确定性 MTE 故障、无法与 flaky 区分」——修正轮定位出**四个真实代码 bug**（均非设备问题），修复后 8 发射全链贯通、对拍 ratio 1.0000（3 seeds × 2 rounds）：

1. **staging 缓冲区 2 倍 undersize（ftail phase-2 故障的真凶）**：驱动 `_bufs` 把 c/kq/vq 分配为 `[t,h,128,16]`/`[t,h,16,128]`（2048 元素/实例），而 fgemv phases 2-4 的 tiling 是 N=32（MatmulApiTiling 拒绝 N=16）、kernel 按 `[128,32]`（4096 元素/实例）寻址——全部实例越界写。越界落点决定是否踩到未映射页（MTE invalid-GM-address）还是只污染相邻张量（silent corruption/NaN），这正是「故障点在 t1/t2 间漂移、standalone 与链内不一致」的来源。修：c/kq/vq 分配 `[t,h,128,32]`，q0_pad 补到 `[128,32]`。最小 repro：`tests/debug_ftail_repro.py old|new`（old=NaN/0.157 错误，new=PASS 0.0023）。
2. **fprep 漏 p**：staging 低秩项重建为 qᵀu，契约是 p·qᵀu（diag 行已乘 s_v·p，q 行没乘）。p=Πexp(g) 逐头差异大（测试轨迹 0.2~5），漏 p 造成 ~14% 误差。修：q 行经 fp32 乘 pval 再转 fp16 写入 A 行 0-3，与 torch 参考舍入路径对齐。修复后 rel err 0.137→0.005。
3. **ftail sumAbs 未清零**：E 累加器 `sumAbs`（别名 scrUb）未初始化就累加（v1 `gdn_lq_flush.cpp:400` 有 `Duplicate(sumAbsUb,0,F_K)`，ftail 抄漏了）→ sk/sv 间歇 NaN（UB 残留字节解释）而 codes 因 Maxs/Mins 把 NaN clamp 成 ±127 反而存活——与观察到的「sk/sv NaN、codes 有限」完全吻合。修：`Duplicate(sumAbs, 0.0f, F_K)`。另在 payload 组装前加 `PipeBarrier<PIPE_ALL>()`。
4. **aclnn executor 池 phase 碰撞（最难缠的间歇性 corruption）**：运行时 executor 池键 = op 类型 + 张量 desc，**不含 attr**——fgemv 的 4 个 phase（ftail 3 个）executor 句柄相同、后建覆盖先建，缓存的 g1 在 g2 创建后实际跑 g2 的 N=32 tiling（输出全零），且表现随 allocator 布局间歇变化（曾误判为设备 flaky / Event 问题）。验证：in-flight 任务的 launch 在 enqueue 时已快照，不受后续建 executor 影响。修：多 phase op **每次发射前现建 executor**（get_ws+launch+destroy，实测 ~0.37ms/级，host 侧），fprep 单 phase 保持缓存。附带发现：`ctx.destroy()` 后 executor 的 desc 悬空，缓存路径必须保活 ctx。

## flush v2 延迟表（贯通后实测，`tests/test_flush_v2.py`）

| B | all-due | nothing-due（skip 步 host wall 中位） | 16 步窗口对账 |
|---|---------|----------------------------------------|----------------|
| 16 | 3.9–4.6 ms | 0.065 ms（max 0.10） | 16 步内恰好 1 次全量发射（step 16）+ 1 次 refresh |
| 64 | 8.8–9.0 ms | 0.065 ms（max 0.10） | 同上 |
| 256 | 27–28 ms | 0.065 ms（max 0.10） | 同上 |

- **due 门控（A2 要求的 host 同步修正）**：`flags.sum().item()` 已删除。fprep 把每个 head 的 fill 值写进 flags → 异步 D2H 到 pinned → host 从**上一步**的 pinned 值模拟 fill（decode 每步 +1、fold 后 wrap 到 1、resync 归 0），预测 due 才发射剩余 7 级；skip 步纯 host（numpy 判定，~65µs）**零 device 工作、零 synchronize**。复制完成度由 device 往返 seq number 证明（`Event.query()` 在本栈恒 False 且 record 有副作用，弃用）。序列数轮替双缓冲。注意 `Tensor.to(dst, non_blocking=True)` 在本 torch_npu 是**静默 no-op**，必须用 `dst.copy_(src, non_blocking=True)`（autogen 陷阱实证）。
- **16 步对账**：从 fill=0 起跑，16 步 decode+flush 中恰好 step 16 一次全量 due（window16 实测 heavy=[refresh, step16]）；64 步 4 窗 = 4 次 due + 1 次 refresh，与预算一致。
- **all-due 27ms@B=256 仍未达个位数目标**：瓶颈是 8192 实例 × 4 phase 的 per-instance 微 GEMM（[128,192]×[192,128]）cube 利用率低 + staging 流量；改进方向是把多实例 batch 成 grouped GEMM（kernel 级工作，未做）。即便 27ms，摊销 27/16×24 ≈ 40ms/step 仍好于 v1 的 66ms/step，且 nothing-due 从 v1 的 0.05ms 再降到 ~0。
- 对拍判据：decoded-state 误差 vs 纯 fp32 dense 参考，kernel/torch ≤ 1.01；实测 **ratio 1.0000**（3 seeds × 2 rounds，含 16 步混合发射模式）。测试脚手架的 simulate_true 已修正（原实现窗口重复折叠 + 允许正 decay 导致参考自身爆炸）。

## 生产路径声明

v2 仍**未接入 gdn.py**（`flush_pools_v2` 仅测试调用）；生产 flush = `gdn_lq_flush`（v1，AscendC 单 op）不变。接入前待办：all-due 降到个位数 ms（grouped GEMM）、 due 门的「每 decode 步恰好一次 flush 调用」不变式在 gdn.py 混合 batch 路径复核。
