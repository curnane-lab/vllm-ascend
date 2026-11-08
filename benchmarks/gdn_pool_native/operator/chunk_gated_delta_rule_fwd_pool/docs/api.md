# ChunkGatedDeltaRuleFwdPool 接口契约

## 参考与范围

- 数学与计算顺序参考：`chunk_gated_delta_rule_fwd` 的 Phase6 A2 实现，固定源码版本 `a4a7958`。
- 本入口新增 pool 的 indexed I/O 与 `[V,K]` 状态物理布局，不改变 Q/K 归一化、KKT、solve、W/U、H 递归或 O 的数学路径。
- 原 `ChunkGatedDeltaRuleFwd` 的公开入口、参数、返回和默认行为保持不变。
- 首版只注册 `ascend910b`。A3 的复用和 A5 均不在已验证支持范围内；不自动启用 A5。

## 公开入口

Python stable adapter 暂定：

```text
npu_chunk_gated_delta_rule_fwd_pool(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
    Tensor pool_state, Tensor source_slots, Tensor destination_slots,
    Tensor has_initial, Tensor? cu_seqlens, Tensor? chunk_indices,
    float scale, int chunk_size=64, int stream=0
) -> Tensor
```

返回值为 sequence-major attention 输出。最终状态直接写入 `pool_state` 的 destination slots，不额外分配或返回紧凑 final-state tensor。`source_slots`、`destination_slots` 和 `has_initial` 在设备上读取，capture 时不做 device-to-host 同步。

aclnn/L0 层使用 `ChunkGatedDeltaRuleFwdPool`，不修改 `ChunkGatedDeltaRuleFwd` 的函数签名。L0 所需 `beta_bht`、`raw_g_bht`、`a_storage`、`g_cumsum_bth` 和 `A` 沿用 Phase6 的内部计算契约；Python 调用者不直接提供这些中间量。pool 专属 L0 attr `pool_slot_stride` 和 `pool_head_stride` 为 INT64 element stride，分别由 descriptor 的 `GetViewStrides()[0]` 和 `[1]` 传入；`pool_head_stride >= K*V`、`pool_slot_stride >= HV*pool_head_stride`，允许 slot/head padding。不能从 optional-input stride API 推导，该 API 在此路径不提供有效 view stride。原 Phase6 无这两个 attr。

## 输入与布局

| 输入 | 契约 |
| --- | --- |
| q/k | `[B,HK,T,128]`，BF16，BNSD；与生产 A2 路径一致，在进入本入口前完成一次 L2 norm。 |
| v | `[B,HV,T,128]`，BF16，BNSD；`HV % HK == 0`。 |
| g | `[B,T,HV]`，FP32，自然对数门控，与生产 `use_exp2=false` 一致；host 转为内部 BHT。 |
| beta | `[B,T,HV]`，BF16；host 转为 Phase6 所需的 FP32 BHT。 |
| pool_state | `[P,HV,128,128]`，BF16，逻辑布局 `[V,K]`。最后两维连续，dim0/dim1 可有 padding stride，且 `stride(0) >= HV*stride(1)`、`stride(1) >= 128*128`；host/kernel 分别传递这两个 element stride。 |
| source_slots | `[N]`，设备 INT32，每条原始序列一个 pool row。 |
| destination_slots | `[N]`，设备 INT32，每条原始序列一个 pool row。 |
| has_initial | `[N]`，设备 BOOL；为 false 时初始递归状态为零，不读取对应 source row。 |
| cu_seqlens/chunk_indices | 与 Phase6 相同的 host metadata；varlen 时成对提供，canonical sequence-major chunk 顺序。 |
| scale/chunk_size | 与生产一致；首版 `chunk_size=64`。 |

固定长度时 `N=B`；varlen 时 `N=len(cu_seqlens)-1`，包括零长度原始序列。空序列没有递归任务，也不写 destination。slot 值必须位于 `[0,P)`；device metadata 的内容有效性由调用方保证，host 不为检查它引入同步。

## 输出、别名与状态语义

- attention 输出为 `[B,T,HV,128]`、BF16、BSND，数学内容与生产 FLA 相同。
- L0 的 `final_state` 输出与 `pool_state` 使用同一基址及 layout，不执行 `Contiguous` 或 `ViewCopy`。
- 非空序列的初始状态为 `has_initial ? pool[source] : 0`。首 chunk 在 UB 内把 `[V,K]` 转成递归所需的 `[K,V]`；中间 H 状态始终保持原布局。
- 最后一个 chunk 的 `[K,V]` 结果在 UB 内转回 `[V,K]`，写 `pool[destination]`。除布局搬运外，FP32 累加、cast 时机和输出顺序与 Phase6 相同。
- 允许 `source==destination`：先完成初始状态读取，最终状态只在末 chunk 写回。
- `source!=destination` 时 source 保持不变；未选择 slot 和 pool 行/head padding 保持不变。destination slots 必须互不重复。
- 错误不得在可能已写 pool 后静默回退到另一条算子路径；调用方收到错误。

## 校验与回退边界

首版只接受上述 BF16、BNSD、K=V=128、chunk64、无 PCP、无核内 Q/K L2 norm、`use_exp2=false` 的生产组合。FP16/FP32 pool、V256、非法 metadata、缺失 pool/index、非连续末两维、其他 layout 或不支持的 SoC 明确拒绝，由 vLLM-Ascend 外层回退现有生产 FLA。

默认路径必须与修改前 Phase6 保持 bitwise；pool 路径与“生产 FLA + gather/transpose/scatter”基准在输出、H/V-new、最终状态、source 和未选择 slot 上要求 `rtol=atol=0`。

## 性能评价

目标场景为 Qwen3.5-4B、910B3、`HK=16/HV=32/K=V=128`、384-token prefill，比较生产 FLA 与 pool I/O，再比较 pool layout fusion。单算子采用 `msprof op_summary Task Duration(us)`，服务采用同 prompt、同 decode 长度、正反启动顺序各 5 轮的 TTFT/wall 中位数；eager 与 ACL graph 分开报告。收益必须以实测为准，不由减少的 gather/scatter 字节数推断。
