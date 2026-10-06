# M3：flush 图内化 结果报告

日期：2026-10-06 ｜ vllm-ascend 提交链：`a9d4110`(M3-1) → `b84e86e`(M3-2) → `d3b9950`(binding 修复)
（M3-3 尝试过、实测回退，未提交，见下）

## 子项完成状态

| 子项 | 状态 | 证据 |
|---|---|---|
| M3-1 fill=L 语义 | ✅ 完成（三重门全绿） | 5 对拍 PASS + v2 parity PASS + 48-token 冒烟 3/4（分歧仍在 token 36） |
| M3-2 flush due 扫描 | ✅ 完成 | nothing-due 0.046/0.078/0.051ms @B=16/64/256（≤0.1ms 达标）；all-due 不退化 |
| M3-3a flush S→fp16 | ❌ 实测回退 | 见下（更慢 + 判据余量变小，已 checkout） |
| M3-3b flush 三段式 cube | ⏸ 未动工（设计已定，工作量估算见下） | — |
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
3. all-due flush 44ms@B=256 未优化（M3-3b 未做）。
4. FULL-graph capture 未复测；piecewise-only 限制维持原结论。
5. leapquant-npu 侧 binding（gdn_lq_*.py）的 warm_cache/dummy 几何与 key 策略同 v1 隐患已修；leapquant-npu 非 git 仓，变更以文件为准。
