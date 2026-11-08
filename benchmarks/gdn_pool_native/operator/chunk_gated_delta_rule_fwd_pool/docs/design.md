# ChunkGatedDeltaRuleFwdPool 设计

方案设计规则版本：V2

## 1. 目标

接口唯一事实来源为 `docs/api.md`。新入口复用固定版本 Phase6 的完整计算图，只扩展状态输入和最终状态输出：

```text
Q/K/V/g/beta -> KKT -> solve -> W/U -> H recurrence -> O
                                      ^             |
                           indexed pool source      indexed pool destination
```

首版模型 case 为 B 条变长序列、总 token 384、HK=16、HV=32、K=V=128、BT=64、BF16、910B3。Phase6 原有 task 映射、AIC/AIV 数量、ping/pong 深度、workspace 和 H/O 同步不变；pool 只改变 state 基址计算和首末 chunk 的 UB 布局转换。

| Stage | 执行单元 | 功能 | 与生产 Phase6 的差异 |
| --- | --- | --- | --- |
| 0 | Cube/Vector | KKT 与 cumsum | 无 |
| 1 | Cube/Vector | triangular solve | 无 |
| 2 | Cube/Vector | W/U recompute | 无 |
| 3 | Vector | 初始化 H | source slot 寻址；fresh 置零；`[V,K] -> [K,V]` |
| 4 | Cube/Vector | V-new 与 H 递归 | 使用原 `[K,V]` H，数学不变 |
| 5 | Vector | final state | destination slot 寻址；`[K,V] -> [V,K]` |
| 6 | Cube/Vector | attention O | 无 |

性能基线为相同输入下生产 `chunk_gated_delta_rule_fwd` 加外部 gather/transpose/scatter。对比对象还包括 sgl direct pool，但不以后者为数值基准。优化空间是消去每条序列、每层两次完整状态 compact 拷贝；代价是首末 chunk 的 UB 转置和 strided pool 写回。只有实际 profile 证明净收益时才采用。

## 2. Stage 0–6 详设

### 公共任务域、地址与符号

`Nbase` 为有效原始序列数，H 的任务沿 `HV` 和 sequence 展开。varlen 的 compact batch 只用于 token/chunk，新增 `stateBatchIdx` 保留原始序列号，保证空序列前后的 slot 不串位。pool 地址按 element 计：

```text
source_base = source_slots[stateBatchIdx] * poolSlotStride + hv * poolHeadStride
final_base  = destination_slots[stateBatchIdx] * poolSlotStride + hv * poolHeadStride
```

`poolHeadStride >= V*K=16384`，K/V 连续；`poolSlotStride >= HV*poolHeadStride`，允许 slot/head padding，不能假定 slot stride 等于 `HV*poolHeadStride`。默认路径继续使用 `(compactBatch*HV+hv)*K*V`，且不读取 slot 数组。

GM/workspace 仍由 Phase6 host tiling 分配，pool 不增加 H、V-new、solve 或 matrix workspace。H tiling 尾部仅为新算子增加 `poolHeadStride:int64` 和 `poolSlotStride:int64`；原算子的 tiling 布局不变。source、destination、has_initial 只增加 GM 只读访问，不写 index metadata。

### Stage 0–2：原 Phase6 前缀

KKT、solve 与 W/U 的公式、任务映射、精度、L1/L0/UB/GM 区间和同步全部沿用 `chunk_gated_delta_rule_fwd`。这些 Stage 不访问 pool，也不因新入口重排计算或改变 cast。

### Stage 3：Vector，初始化 H

输入为 source pool 或零状态，输出为原 H workspace 的 `[K,V]` 首 chunk。AIV0/AIV1 仍分别处理 K 行的两个不重叠半片，每半片 64 行；每核任务分配不变。

`initialStateOffset` 已包含 slot/head 基址。标量正确性基线读取 `source[v*K+k]`，写入紧凑的 UB `[K,V]` 半片；head 内行 stride 必须为 `K`，不能复用 `poolHeadStride`。该路径不引入 GM gather/scatter，也不改变数学计算，已通过 17 组生产对照。

```text
UB state[0,32KiB) / pong[96,128KiB)  输入[V,64]，标量转置读取
UB h[64,96KiB) / pong[160,192KiB)    输出[64,V]，标量转置写入，MTE3 末次读取
```

目标线性下标 `i=k*V+v` 对应源下标 `v*64+k`。标量写 UB 后通过 S→MTE3 event 同步，再由 MTE3 按目标行 stride 写回；S→V 只覆盖 Scalar→Vector，不能代替这个发布动作。fresh 序列走原 `Duplicate(0)`，不读取 source。下一次复用 UB 前等待 MTE3。Gather 与 tile 优化均未通过正确性或时限验收，不进入服务实现。

`src==dst` 时本 Stage 在 final 写回之前完成；独立 destination 不写 source。无效 row 不产生任务，两个 AIV 的 CrossCore 通知保持配对。

### Stage 4：Cube/Vector，原 H 递归

沿用 Phase6 `W@H`、`K^T@V-work`、gate/cast 和 H/O fused scheduler。H 的 GM 布局、FP32 update、L1/L0C 和 ping/pong event 不变；因此 pool 不得改变输出或递归状态的数值。新增的 pool 参数不参与 gemm 形状或矩阵地址。

保持 Phase6 原有 wave、V1/V2 和 event 生命周期。生产 Stable-ABI 默认 no-wait 调用的不稳定性已在后端对照中复现并通过强制 stream drain 消除；pool 的地址修复不引入对 H/V-new 调度的改动。

### Stage 5：Vector，写回 destination

末 chunk 的每个 AIV 半片先按原顺序形成 FP32 `[64,V]` update，再按原 `CAST_RINT` 转为 BF16。标量正确性基线在 UB 内转置后，以 `rows=V, cols=K-half, dstStride=K` 批量写入已定位的 destination head。

```text
UB calc[0,32KiB)        转置后的[V,64] BF16，占用16KiB，不覆盖仍需要的 FP32 update
UB hUpdate[32,64KiB)    原 FP32 [64,V] update，直到 cast 完成
UB h[80,96KiB)          cast 后[K=64,V] BF16，转置输入
pong 使用对应的 +96KiB 区间
```

源矩阵为 `[K-half,V]`，目标为 `[V,K-half]`。设 `rows=K-half`、`cols=V`，目标线性下标 `i=v*rows+k` 对应源下标 `(i%rows)*cols+i/rows=k*V+v`，所有下标均限制在源、目标的 `rows*cols` 个元素内。标量转置前先以 V→S event 保证 Vector cast 对 Scalar 可见；写入完成后再通过 S→MTE3 event 发布给 MTE3，以 `rows=V, cols=K-half, dstStride=K` 写 destination。scheduler 已把 destination head 的基址计入 `finalState`；`poolHeadStride` 只用于不同 head 的基址寻址，绝不能作为 head 内 `[V,K]` 的行 stride。原 `V_MTE2`、`V_MTE3`、`MTE3_MTE2`、`MTE3_V` flag 在相同资源边界设置；转置插入 cast 和 MTE3 之间，不放宽原有同步。

非末 chunk 完全不进入 pool 写回。首版 K=V=128、V 不分块，因此不会进入原 V256 fallback 路径；host 拒绝未覆盖的 wide-V/tail 形状。

### Stage 6：Cube/Vector，原 O

沿用 Phase6 的 Q/K/V-new/H 输入、gating、scale 和 BSND 输出转换。pool 状态写回与 O 的依赖仍由原 H/O 同步保护，不改变 output token 的数学结果。

### 搬运、同步与回退汇总

| 数据 | 原 compact 路径 | pool 路径 |
| --- | --- | --- |
| source state | 完整 gather + transpose 后供 H 读取 | 首 chunk 每行读一次，UB 转置 |
| destination state | final transpose + 完整 scatter | 末 chunk 每行写一次，UB 转置 |
| index metadata | host/device gather/scatter 索引 | H scheduler 只读 src/dst/has_initial |
| H/workspace | 原有 | 原有，无额外 GM workspace |
| pool padding | gather/scatter 外围处理 | 不读不写 |

默认路径不读取新增输入，所有原 flag、workspace、同步和返回顺序保持不变。任何不支持组合在 host 或 vLLM-Ascend gate 拒绝，不在部分写 pool 后自动重算。
