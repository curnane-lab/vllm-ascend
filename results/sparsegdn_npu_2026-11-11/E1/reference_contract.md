# SparseGDN E1 独立参考合同

## 来源与范围

唯一算法来源为 [SparseGDN v1, arXiv:2610.41420](https://arxiv.org/html/2610.41420v1) 的 §2.1–2.7、§3.1、§3.5–3.7。2026-11-11 重新检索论文页面、arXiv API、GitHub 和 SGLang/vLLM/vLLM-Ascend/FLA 同名实现，未找到作者公开仓库或同名 PR。本实现是按论文独立实现，不声称官方复现；搜索无结果不排除未公开实现。

本阶段只研究算法参考和输入分块语义，没有 NPU kernel、性能实测或真实模型质量结论。实现位于 `experiments/sparsegdn_npu/sparsegdn_reference.py`，执行入口为 `test_sparsegdn_reference.py` 与 `test_sparsegdn_partition.py`。

## 数值与布局

- 高精度参考使用 CPU FP64。Q/K 在输入前已逐 head L2 归一化，参考内部不重复归一化。
- Q/K 为 `[T, 16, K]`，V/Z 为 `[T, 32, V]`；value head `h` 使用 QK head `h // 2`。两个相邻 value heads 共享 mask，但各自持有独立状态。
- 论文状态在本参考中为 `S[K,V]`。生产 cache 的 `[N,H,V,K]` 布局必须在接线时转置，不能按字节直接解释。
- `log_decay[T,H] <= 0` 是每 token 的 log2 decay。每个 64-token chunk 内重新计算 `g = cumsum(log_decay)`；`2^g` 是 retention，不能改用自然指数或重复 cumsum。
- `beta[T,H]` 同时进入 Delta 矩阵及 U/W 两个右侧项。W 的源 token 权重为 `beta * 2^g`，U 的源 token 权重仅为 `beta`。
- 初始状态为零是论文 cold-prefill 合同。非零 `initial_state` 仅用于 dense 恒等和 prefix 诊断，是实验扩展，不构成论文支持非零 h0 的证据。

## Delta solve 与近似组

R0 是独立逐 token 递推：

1. `read = K[t] @ S`。
2. `S = 2^log_decay * S + beta * outer(K[t], V[t] - 2^log_decay * read)`。
3. `O[t] = Q[t] @ S / sqrt(K)`。

R1 使用完整 64×64 Delta solve、全事件 mask 和论文 group-2 输出。R1 的每个有效 token 输出和最终状态必须对 R0 达到 `rel-F <= 1e-8`，参考绝对值小于 `1e-9` 的位置另要求 `max abs <= 1e-9`。

A1 不计算完整 inverse 后 mask。它对四个 16×16 对角块分别求逆，再按论文式 (6) 构造三个直接相邻块。U 逐块生成；七个 `C_bar` 块保存后才按事件生成 W。A2 使用 Full64 solve 和独立重算的 adaptive mask。A2 与 S-ref 的 mask 差异不能全部归因于 solve。

## Selector 合同

- 有效 chunk 数为 `C = ceil(T/64)`；padding 不参与排序、预算或 gate 采样。
- Tail-16 score 是 U 最后 16 行的 Frobenius norm。每个 value head 单独按整段请求的 L2 norm 归一化，再对 head pair 取平均；全零 score 向量归一化为零。
- tiers 为 `(0.08, 0.30, 0.95)`；`C >= 2` 时 `k = clip(round(rho*C), 2, C)`，Python/FP64 round 为 ties-to-even；`C=1` 时唯一 chunk 强制保留。
- 首、尾 chunk 是 anchors；非 anchors 按 score 降序、chunk 索引升序打破平局。三档候选集使用同一排名，因而嵌套。
- lifetime 使用最后最多 16 个有效 chunk 的平均 chunk-end retention，并计算长度为 `C-2` 的截断几何和。
- Z 在每个 16-token block 的偏移 8 采样，坐标为 `73*(4*c+a) mod V`；使用 `norm_scale * SiLU(Z)` 的平方能量，越界样本为零。所有 value heads 的平均能量为零时，visibility 全零。
- upgrade gain 使用未归一化的 Tail score、lifetime 和 visibility；两个 gain 向量分别按 16 对的均值归一化，均值为零时映射为零。
- 先选 1–4 个 high pairs，再按论文表选 medium pairs。CSR 行按 high/medium/low、再按物理 pair 索引排序；事件按时间升序。
- FP64 参考对 split score 的“平局”使用精确相等；平局时先取离 fallback 最近的 count，再取较小 count。完全平坦的 gain 使用 high=3、medium=`4*(4-high)`。论文未给出额外的近似相等容差，本参考不擅自加入。
- `group_event_map` 存储 CSR 的全局 packed event index，而不是行内序号；没有组前事件时为 -1。`decay_prefix` 同时包含选中和跳过 chunk。
- adaptive sparse 的 W-packed capacity 为 `16 * kH`，实际事件数为设备内概念上的 E；本 CPU 参考只验证 `E <= capacity`，不把容量误写成实际 keep。

## 跳过、输出和 padding

选中事件计算 `U' = U - W @ S` 并写内容；跳过事件只保留 decay，`U' = U`，不生成 W，也不做内容写入或 recurrent read correction。

输出按 `(2r, 2r+1)` 两 chunk 为一组。奇数 chunk 始终包含相邻偶数 chunk 的局部贡献，即使偶数 chunk 被跳过；该贡献不进入持久 state，也不参与奇数 chunk 的 recurrent correction。式 (25) 的补偿项有专门非零测试。

尾部补齐为 Q/K/V/beta=0、log_decay=0（单步 retention=1）、Z 越界样本为零。奇数 chunk 数的最后一个组只返回真实 token；所有有效 token 输出均以 poison/NaN 初始化后覆盖检查。

## 分块诊断的两条政策

1. **Adaptive partition**：每个 scheduler piece 独立重算 selector。它用于测量算法/policy 是否随切分变化，不是把该变体冒充论文全局选择。
2. **Frozen-mask partition**：先固定一次性输入的 mask/tier，再按 64/128/256/1024/2048-token pieces 执行。偶数 chunk 的 corrected U、Q/K/gate 和 group-entry state 被保留到奇数 chunk 到达；这条路径只验证已固定算法选择后的状态与 group-2 重建。

非 64 对齐的跨调用续算需要保留部分 chunk 的 Delta solve、cumsum 和选择上下文。本参考拒绝把 suffix 重置为新 chunk 后称为有效续算，相关用例标为 `BLOCKED-CHUNK-ALIGNMENT`。

## 数值防护与实现修正记录

Delta/local 矩阵先在掩码内保留指数、掩码外置零，再计算 exp2 并应用数值 mask；不得先对未来位置的大正指数求 exp，再以 `0 × Inf` 形成 NaN。批量驱动只对选中 heads 生成 W，并保留与慢参考相同的 CSR/group-2 定义。

初版 deferred-W 的相邻块索引错误和上述掩码溢出问题均已通过独立对拍定位并修复。`vectorized_reference.json` 包含全部 W block、慢/快驱动 mask/U/O/state 对齐及大 decay 回归。修复前工件保留在 `raw/invalid_before_w_fix/` 和 `raw/before_mask_overflow_fix/`，不作为判决依据；真实 layer 0 的 NaN 最小输入保留在 `/tmp/sparsegdn_debug_trace/`。

## 已执行验证

`dense_identity.json` 覆盖 1–5 个 chunks、非整 64 尾部、非零 h0 的 dense 诊断，以及 K=V=128、16 QK/32 V heads 的真实几何。`approximation_cases.json` 覆盖 R1/A1/A2/S-ref、block support、Tail/预算/anchors/ties、CSR/group map、零输入、group-2 补偿和重复执行。

`prefix_partition_small.json`、`prefix_partition_real.json` 及边界用例检查相同 P 配不同 X/Y、独立 P、一次性 P+X/Y、scheduler pieces 和私有槽恢复。JSON 中的 proxy logits 只是固定玩具投影，不能称为完整模型 logits。

真实模型 4K/8K 复核已完成，见 `model_scopea_4k.json`、`model_scopea_8k.json` 及 `raw/model4k/`、`raw/model8k/`。两次各三次完整请求均让实际 sparse 输出驱动后续层，并保存逐层 mask/tier、U/O、边界/最终 state 和完整最后 logits。计算为 CPU FP32，文件中 U/O 为 BF16、state/logits 为 FP32；这些结果只用于算法语义，不是质量或性能验收。4K/8K 均出现真实的前缀 mask/state/O 与最终 logits 差异，进一步支持 SCOPE-B 阻塞。
