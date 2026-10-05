# 实测结果摘要（Ascend 910，Qwen3.5-9B，2026-10-05）

## 精度

| 验证项 | 结果 |
|---|---|
| pools vs 稠密 fp32 oracle（合成，96 token / 6 窗口） | 输出 rel err first16 max 0.0033 / last16 max 0.0055；flush 后状态重建 err 0.0043–0.0053 |
| 真实权重 dump 回放（torch oracle vs fla_npu AscendC 稠密 kernel） | dense-replay 0.15%（约定确认）；**LeapQuant vs 稠密逐步输出 0.05%–0.19%** |
| 端到端贪心生成 48 token（4 prompts，mode 1 torch 路径） | 4/4 与 fp32 基线逐 token 一致 |
| 端到端 768 token（2 prompts，mode 1） | 27 / 138 token 处近tie翻点后各自延续 |
| prompt-logprob drift（3.2K token 重复文本） | LeapQuant 与 fp32 两次运行之一**逐值相同**（mean\|Δ\|=0.131 即该硬件运行间噪声底） |

## 性能（微基准，ms/step/layer）

| B | torch step | triton step | 加速 |
|---|---|---|---|
| 16 | 6.914 | 0.456 | 15.2× |
| 64 | 8.860 | 1.233 | 7.2× |
| 256 | 16.403 | 4.339 | 3.8× |

torch flush（未 kernel 化）：B=16 时约 13–21 ms/层/step，为当前主要瓶颈。
全模型 4 prompts × 48 token：fp32 基线 3.8s，LeapQuant mode 2 16.9s（flush 主导）。

## 调试记录（三个关键 bug）

1. **死槽位 NaN**：窗口缓冲死槽位（j≥fill）残留旧 fp32 状态字节，按 bf16 重解释可为
   NaN，`0×NaN=NaN` 毒化状态重建。修复：decode/flush 加 live-mask，resync 清零缓冲。
2. **recurrence 约定**：fla 语义 = 先衰减后 delta 校正
   （`h*=exp(g); v-=h·k; v*=β; h+=v⊗k`），与 PR 参考一致；由双层对拍确认。
3. **prefill 状态索引为 per-token**（mm warmup 1600+ 条），resync 必须按 slot 去重；
   dump 对比时稠密 kernel 原地更新状态，初始状态必须在 kernel 调用前捕获。

## triton flush kernel 状态

三轮迭代：全 tile 版（UB 需 486KB > 192KB 上限）→ V-chunk 六遍扫描版（UB 收敛）
→ 动态循环 + 谓词化存储（排除早退死锁）。最终卡在 **BiSheng 编译器挂起**
（>10 min 无响应）。结论：flush 转 AscendC 实现（与 sketchssm-npu ASCENDC_PLAN 一致）。

## 更新 2026-10-05（mode 3 = triton step + AscendC flush）

- e2e 冒烟 4 prompts x 48 token：**4.8s vs fp32 基线 3.8s**（torch 路径 77s，mode 2 16.9s）；
  3/4 prompt 与基线逐 token 一致，1 条在第 36 token 近tie翻点。
- 端到端 AscendC flush 集成中修掉的两个关键问题：
  1. autogen aclnn wrapper 的隐藏 contiguous 暂存（任何输入带 AutoContiguous 即触发）——
     strided KV-pool slot 布局下地址错位 + in-place 别名失效 → op def 全部改 IgnoreContiguous；
     注意只改 def 不会触发 autogen 再生，须删 build/autogen/aclnn_gdn_lq_flush.* 与对应 .o 再构建。
  2. autogen 按 state 大小派生 workspace（~2.3GB/executor，24 层 OOM）→ 全 executor 共享一个 ws。
- 长上下文状态误差曲线（真实权重 replay，1080 token / 67 窗口，论文图 3 的 NPU 复现）：
  checkpoint 重建 rel err 首 8 窗口均值 → 末 8 窗口均值：layer 0: 0.14%→0.32%（2.3x），
  layer 8: 0.44%→1.52%（3.5x），layer 16: 0.48%→1.35%，layer 22: 0.35%→1.02%。
  误差增长 ~3x 后饱和（per-window 量化的收缩映射特性），非 per-step 量化的线性发散。
  数据：results/long_rollout_curve.json（gdn.py dump 设施 + scripts/replay_long.py 回放）。

## 更新 2026-10-05（e2e 吞吐基准，enforce_eager，128 tok/prompt，greedy）

| batch | fp32 基线 tok/s | mode 3 tok/s | 比值 |
|---|---|---|---|
| 16  | 203.8  | 168.2  | 0.83× |
| 64  | 701.8  | 470.7  | 0.67× |
| 256 | 1844.9 | 921.6  | 0.50× |

结论：kernel 级流量优势尚未转成端到端收益。每步每层分量（B=256）：
triton step 4.34ms（有效带宽仅 ~1.1TB/s，tl.sum 延迟墙）+ AscendC flush 摊薄
~2.8ms + eager launch 开销，vs 稠密 kernel ~5.8ms。优化方向（进行中）：
step 移植 AscendC（目标 ≤2ms/层 @B=256）、flush 行块 ILP、due-list 共享、ACL graph。

## 更新 2026-10-05 晚（step op + 两 bug 根因 + ACL graph 首验）

### 两个生产 bug 的根因与修复（跨 AI 评审定位）
1. **flush u 写回**：ILP 重构后 w/p 重置的 Duplicate 落在 f32 偏移 4352/4368，
   与 u（字节 17408 起）重叠，踩掉主补偿 token r=0 的前 48 个 fp16 → u rel 31.6%。
   修复：w/p 移到 q 之后（4864/4880），outQ 扩到 19968B（原 18944B 差 512B 溢出）。
2. **step 窗口系数**：(a) w 衰减 `w *= at` 在早期编辑中被静默删除；(b) kn 侧
   redUb[j*8] 与 qn 侧 redUb[64+j*8] 从 j=8 起重叠。修复后 step 对拍全绿。

### 工具链结论（重要）
`WholeReduceSum` 的 repeat-form（repeatTime=16）在本工具链语义不可靠
（dstRepStride 单位是 datablock 且随 arch 变化）——已废弃，改用 repeat=1 单发形态。
代价：step AscendC 0.541/1.563/5.573 ms（0.78× triton），reduce 端口-bound，
暂不接替 triton step（mode 3 保持 triton step + AscendC flush）。

### ACL graph 首验（util 0.80，B=16）
- PIECEWISE 捕获 5/5 通过（mode 3 的 leapquant op 在图内捕获兼容）。
- FULL 捕获完成、replay 功能正确，但吞吐 43.1 vs 基线 215.6 tok/s（0.20×）——
  异常待查（疑似图内 aclnn executor 调用路径）。eager 模式不受影响。

### e2e 最终矩阵（修复后，与回归前一致）
B=16/64/256：mode 3 = 168.4/470.6/921.5 tok/s vs 基线 204.5/703.9/1900.0。

## 更新 2026-10-05 夜（M1 布局重排 + TurboQuant 双源调研）

### M1：per-(slot,head) 连续 32KB 块布局（已合入）
checkpoint 前 19KB（codes 16K + scales 1K + fp16 补偿 token 2K）+ bf16 窗口缓冲 +
计数器，对齐 PR #10 pool 几何。五项对拍全绿（pools/triton step/AscendC flush/
strided/AscendC step）。triton kernel 计时持平（0.444/4.238 ms @B=16/256）——
布局收益要等 M2/M3 的融合式 per-head 读取器才兑现；之前 bench 显示的 5× 回退
是 harness 的 reset 散写开销（strided 标量 scatter），非 kernel 回归。

### TurboQuant 两源调研结论（定位修正依据）
1. **vllm-ascend PR #14715**（厂商 CANN 算子，MLA KV 4-bit）：单请求 decode
   反而慢 25%，收益全在高并发容量场景（2.68× KV 容量 → 并发 2 起 +23%）。
   明确定位"不改善单请求延迟"。→ LeapQuant 卖点同样应是容量/并发。
2. **Ascend-SACT/turboquant-vllm-npu**（社区 triton-ascend 移植）：实证
   **Ascend 910 上 triton-ascend 的 tl.dot INT8 matmul 编译不过**——
   M2 的 cube GEMV 必须走 AscendC MIX，triton 路线放弃（与我们的 AscendC
   reduce 结论互证）。单请求 decode -44%。可借鉴：编译器避坑清单
   （tl.join/reshape 寄存器映射错误→分拆点积；不 autotune；BLOCK 尺寸约束；
   centroid 表保 fp32）、decode kernel 骨架（一 program per (seq,head)、
   block-table 两级查址、SRAM 解压不回写）、per-token 连续字节行布局、
   逐层 cosine≥0.998 验证法。负面教训：norm 绝不存 uint8 截断；
   192KB UB 预算下别贪多字段合一 kernel。
