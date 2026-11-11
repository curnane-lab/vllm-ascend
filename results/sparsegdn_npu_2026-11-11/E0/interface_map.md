# E0 生产 GDN 接口与执行身份

## 已闭合的运行身份

- 本机是 8×910B3 64GB，不是 910C。E0 数字仅为 TP1 开发/交叉验证，不替代 910C 验收，也不外推 TP8/EP。
- API/EngineCore worker 实际加载：vLLM `/vllm-workspace/vllm`（`ced6857afa0ea7b2e3f0846a62e1394e90f15607`）和 Ascend `/vllm-workspace/vllm-ascend`（`c09d3e20b9c45c80b34ae3781440d90904a44023`）。两仓 tracked diff 均为空。
- 生产 FLA/OPP 使用系统包 `/usr/local/python3.12.16/lib/python3.12/site-packages/fla_npu`。系统 `libcust_opapi.so` 同时导出 `aclnnCausalConv1dGetWorkspaceSize` 与 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize`。
- recurrent-only 实验 OPP 不含 causal-conv 符号，不能用于完整模型服务。版本配套错误的 `83f5de4 + 2f253e5` 服务组合也不能冒充当前生产组合。
- 当前 worker 是 `vllm_ascend.worker.worker.NPUWorker` / `model_runner_v1.py`。启动日志证明 7 个 `decode, FULL` 图 capture 完成；prefill 没有宣称 FULL replay。
- 基线使用纯文本模式以避免无关视觉 warmup；BF16 权重/激活，模型配置的 recurrent state dtype 为 FP32。TP1、chunked prefill、`max_num_batched_tokens=1024`、`max_num_seqs=64`、prefix cache 关闭、async scheduling 开启。

## QKV、Z、gate 与输出门

实际调用位于 `/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py`：

1. `forward` 在 `gdn.py:377-450` 计算投影。Qwen3.5 的 fused `in_proj_qkvz` 在 `gdn.py:392-397` 拆成 `mixed_qkv` 和 Z；Z reshape 为 `[T, local_Hv, Dv]`。
2. Q/K/V 经过 conv 后，在 `gdn.py:680` 由 `DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)` 产生 g/beta。这里的 g 是每 token log2 decay，不是已经跨 chunk  cumsum 的值；SparseGDN 参考只在 64-token chunk 内 cumsum 一次。
3. core 前 Q/K 做 L2 norm。CANN wrapper 的合同见 `gdn.py:294-348`：Q/K `[1,T,Nk,Dk]`，V `[1,T,Nv,Dv]`，g/beta `[1,T,Nv]`。
4. core 输出先经过 `self.norm(core_attn_out, z)`（`gdn.py:440-445`），再 reshape 并进入 `out_proj`。因此 selector 所需的真实 Z 可从当前层 projection 获得，RMSNorm scale 为 `self.norm.weight`。现有 core ABI 不接收 Z 或 norm scale，后续原生实现必须扩展输入或在同一层内提供等价统计；不能用常数 visibility 冒充论文 selector。

## State 轴序与 cache 生命周期

- 论文参考状态是 `S[K,V]`；生产 cache/ CANN initial state 是 `[N,Hv,Dv,Dk]`。FLA 的 `state_v_first=True` 对应 V-first 生产布局，参考接线必须显式转置。
- CANN 分支通过 `gather_ssm_states`（`gdn.py:815-820`）读取 `prefill_state_indices` 和 `prefill_has_initial_state`，并在返回时 `scatter_ssm_states_`（`gdn.py:833-837`）。FLA 分支对应 `gdn.py:790-809`。
- Triton fallback 才在 `gdn.py:839-840` 对已有 initial state 调用 `clear_ssm_states`，其 mask 仍是 `prefill_has_initial_state`。不得为迎合论文 zero-state cold prefill，额外清零已有 cache state。
- CANN 文档路径 `gdn.py:330-332` 将 initial state 转为 BF16；这不是允许把 SparseGDN 的 FP32 accumulation 降为 BF16 的依据。候选实现的 coefficient/state 实际 dtype 必须另报。
- `PAD_SLOT_ID=-1`，`NULL_BLOCK_ID=0`，定义于 `/vllm-workspace/vllm/vllm/v1/attention/backends/utils.py:46-47`。conv 调用同时使用 `pad_slot_id=PAD_SLOT_ID` 和 `null_block_id=0`，例如 `gdn.py:627-628`。
- hybrid KV cache 的 state page 可能有 padded batch stride，不能假设 `ssm_state[0].numel()` 就是 slot stride。FLA 的通用非连续 state 合同另有每 head 前置 16 elements padding 的测试（`examples/fast_kernel_launch_example/tests/recurrent_gated_delta_rule/test_recurrent_gated_delta_rule.py:39,121-136`），E3 必须做 guard/64-bit 地址验证。
- Qwen3.5-4B 为 16 QK heads、32 value heads、K=V=128；每个 QK head 对应两个相邻 value heads。模型 32 层中 24 层是 GDN，linear heads/dims 来自实际 `config.json`。

## 实际 prefill 分派

当前 B3 生产 trace 命中 `aclnnChunkGatedDeltaRuleFwd_*`，不是慢 Torch 或旧 FLA 对照。4K 短窗中该 op 共 96 次，即 4 个 1024-token scheduler pieces × 24 GDN layers。trace 还包含 causal-conv、QK L2 norm、gating、state gather/scatter 和最终输出处理。

候选 SCOPE-A 研究 wrapper 同时覆盖 `DeviceOperator.fla_gdn_prefill` 与 CANN 静态 wrapper，并对多请求、非零 h0、mixed decode/prefill、非完整 prompt 和 dense fallback 直接报错。它默认关闭，不参与生产 dense 基线，也不会向共享 prefix state cache 发布中间状态。

## all-fresh 与 cached-prefix

当前正式 B1/B32/B64 基线均关闭 prefix cache，请求是 all-fresh。启用 cache 的 dense smoke 和 32K 补充基线需单独服务配置，不与当前测量混写。SparseGDN 的 SCOPE-A 研究只处理完整 cold prompt，常规 cached-prefix 续算由 E1 语义门单独判决，不能由 dense cache 正确性代替。
