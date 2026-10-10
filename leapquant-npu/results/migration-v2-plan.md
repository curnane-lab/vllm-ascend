# LeapQuant NPU 移植实验计划：GPU 执行路径复现轮

日期：2026-10-10（Asia/Shanghai）。版本：v2，替代 2026-10-09 的容量优先安排。
状态：第一轮已有宿主报告，记在 `M5-measurement-update.md`。本文仍是协议，不是新的配对窗口。计划默认使用 910C 64GB、Qwen3.5-9B。
决策：先执行 R0–R2 的受控探索；R3/R4 由正确性和预算筛选解锁。原报告 E5/E6、step v2 接线、生产量化开关、compact page 和分配器改造继续暂停，直到相应门槛通过。

记录时对照 `M5-measurement-update.md`（主分支 `4e3ded7`）改了两处，避免把门限写反，或把仓库里没有的账写成已测结论：

- 停止线按 `14.527 / 16 = 0.9079` 写。step 平均达到 0.908 ms，即使其余工作免费也进不了门。F 达到 14.527 ms，即使 step 免费也进不了门。
- 显存拆分和 Cube 读出的宿主报告已写入测量笔记。原始 json 与脚本仍不在本仓库。六条 cached-map 是块表的 0.60%；24.8 MiB 只是六个单块，不是八份 storage 的合计。容量线对这两档负载停止。Cube 读出 0.677 ms/步留出 3.70 ms 给 step 余量和 flush，R3/R4 未解锁。

## 1. 本轮要验证什么

本轮不再以“量化存储变小”推断吞吐，也不再在现有串行向量链上无预算地打磨局部原语。要验证的是：能否在 910C 上复现作者 GPU 实现的直接读出、精度处理、矩阵计算和流水组织，让完整窗口超过当前生产 FP32 dense。

四个可证伪假设：

| 假设 | 本轮变化 | 支持它的证据 | 否定后怎么处理 |
|---|---|---|---|
| H1：直接读出配合批量转换/矩阵计算，能避免每步重建 S 的代价 | 同一次读取 codes 完成 key/query 两个读出；补偿项和窗口使用向量点积 | 正确的完整 step 达到窗口分配预算 | 当前组织停止；不以“比旧 step 快”解锁引擎 |
| H2：flush 的主要工作可用矩阵计算和残差/低秩分解降低成本 | 保留旧补偿项的因子形式，矩阵计算负责重建和拟合 | 作者逐 head 精度契约通过，且真实 flush 总成本进入预算 | 该数值/硬件组织停止；不继续削减无关扫描 |
| H3：设备端 due 筛选与均匀分发改善混合 due 尾部 | 以设备 fill 为真值，均匀分配 due head | 真实混合 due 完整窗口改善，非 due 字节不变 | 保留正确基线，不以 host 镜像测量替代 |
| H4：压缩历史 checkpoint 能转化为可用服务容量 | 只先审计现有显存数据；满足空间条件后才改分配 | unique storage、cache 生命周期与请求驻留账自洽 | 容量线保持暂停 |

H1/H2 是当前速度主线。H3 不替代 all-due 工作的优化。H4 不由启动日志的 1.26× 或 1.145× 自动成立。

### 首轮执行范围

1. R0：固化真正运行的源码/二进制；确定可服务的 batch；审计已有记录。
2. R1：解决零输入残留、NaN、跨 pool/idx 绑定的最小可靠性问题。
3. R2：只做两个真实 codes 读出的 Vector/Cube 可行性比较。
4. 每条候选最多两轮实现迭代；进入下一轮必须有正确结果、真实计时和仍可达到完整窗口预算的证据。
5. 不在首轮跑完整模型扫并发、五任务全量、分配器改造或单核 MIX 的长期调试。

## 2. 固定证据、平台与比较基线

### 来源快照

- NPU 仓库：主分支 [4e3ded7ebf5c8e7ab86ff09a854f8375195d848e](https://github.com/curnane-lab/hybird-quant/tree/4e3ded7ebf5c8e7ab86ff09a854f8375195d848e)，核对日期 2026-10-10。
- 当前测量笔记：[M5-measurement-update.md](https://github.com/curnane-lab/hybird-quant/blob/4e3ded7ebf5c8e7ab86ff09a854f8375195d848e/leapquant-npu/results/M5-measurement-update.md)。
- 作者 [NVlabs/kda PR #10](https://github.com/NVlabs/kda/pull/10)，head [48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326](https://github.com/Conless/kda/tree/48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326)，当前为 open。
- 作者 [GDN step 源码](https://github.com/Conless/kda/blob/48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326/requests/conless-leapquant-gdn-step/baseline.py)、[step 契约](https://github.com/Conless/kda/blob/48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326/requests/conless-leapquant-gdn-step/README.md)。
- 作者 [GDN flush 源码](https://github.com/Conless/kda/blob/48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326/requests/conless-leapquant-gdn-flush/baseline.py)、[flush 契约](https://github.com/Conless/kda/blob/48f02e6ba0e98a1841ab6f0a2d99a90b2de1a326/requests/conless-leapquant-gdn-flush/README.md)。
- 原论文：[LeapQuant, arXiv 2609.38166v1](https://arxiv.org/html/2609.38166v1)。

作者 PR 公开的是 B200 算子和部署契约，不包含完整 vLLM 分配器改造。公开 Pool 仍按每 head 64 KiB 区域分配，checkpoint 只使用前 19,456 B，窗口为另外的张量。不能把这份 Pool 当作论文显存压缩已经兑现的代码证据。

### 默认平台

vLLM 0.27.1 / vllm-ascend 0.27.1rc1 `3b3188623`，CANN 9.1.0，torch 2.10.0，torch_npu 2.10.0.post4，triton-ascend 3.2.2。Qwen3.5-9B：24 GDN＋8 FA；HK=16、HV=32、K=V=128；L=16、R=4。以运行时实际版本、芯片 ID、核数和 import 路径为准。

容器补丁、探针、部分脚本尚未完整入仓；上述仓库 SHA 不能单独证明运行环境。

### 当前有效观察与限度

| 观察 | 本轮用法 |
|---|---|
| B=256 生产 dense 完整 16 步窗口为 15.98 ms | 维持其原协议下的 10% 收益门：14.527 ms |
| 旧 eager Torch dense 为 59.15 ms | 只作历史诊断；不作为新的收益对照 |
| 当前 all-due flush 为 24.33 ms | 当前 flush 单项已超完整预算，原组织不能靠空 due 优化过门 |
| add-fold 0.98 µs 对多个输入族通过，但零输入有残留；decode 读回出现 NaN | 原语速度不能代替完整候选正确性 |
| 7.58 µs/实例及 0.50 ms step、18.4 ms flush 的拼装 | 前者是探针；后两者没有真实实施计时，均不作为通过结果 |
| B=2 dense GDN 约占墙钟 4.5% | 小 batch 作为集成检查，不承诺论文量级端到端收益 |
| 96 路吞吐比约 0.54，启动预算有所增加 | 审计完成 token/时间口径；不将速率乘容量倍率 |
| step v2 的跨 pool 状态读写不一致 | 保持关闭，不作为新 Cube 方案的默认可靠入口 |
| 用户侧称为 §26 的显存拆分 | 本仓库测量笔记没有这份原始账。先找文件再审计；找不到记缺口，不据此写“少量 cached 条目” |

### 完整窗口预算

主门只接受真实配对窗口：

`T_LQ = 16 次实际 step + 所有实际 flush 调用 + 必要转换/搬运/运行时成本`。

若采用每步固定调用 flush，15 次 nothing-due 也在窗口内。若采用安全跳过，则跳过判断及生产必要的更新也在窗口内。组件估算 `16*t_step + t_all_due` 不计空调用成本，不能作为通过结果。

在原 B=256 对齐范围下，要求 `T_LQ <= 15.98/1.10 = 14.527 ms`。下表的 F 是整个窗口中除 step 外的必要总成本，不仅是一次 fold：

| step 平均时间 | 16 次 step | F 可用上限 |
|---:|---:|---:|
| 0.30 ms | 4.80 ms | 9.727 ms |
| 0.40 ms | 6.40 ms | 8.127 ms |
| 0.50 ms | 8.00 ms | 6.527 ms |
| 0.80 ms | 12.80 ms | 1.727 ms |

0.40 ms＋8.0 ms 是初始组合目标，不是硬件预测；优先争取更多波动余量。停止线：`14.527 / 16 = 0.9079`。step 平均达到 0.908 ms 时，即使其余工作免费也无法进门；F 达到 14.527 ms 时，即使 step 免费也无法进门。它们用于停止某个组织，不代表硬件理论极限。

B=128 的历史 D_op 为 8.176 ms，对应同协议门约 7.433 ms。新协议、新 batch、新 layout 或更快 dense 需要独立配对对照。保留 B=256 原门；不通过更换弱基线降低要求。

## 3. 通用 harness、精度与数据产物

新增脚本名均为待实现建议入口，不能视为仓库现有可执行命令。建议位置：`leapquant-npu/tests/`、`leapquant-npu/scripts/`。运行目录：`results/migration_v2/<R编号>/<run_id>/`。

每项提交以下工件：

- `manifest.json`：代码与 dirty diff hash、镜像 digest、编译参数、真实 import/so 路径、二进制 SHA256、模型 revision、芯片/核数、环境开关、shape/stride/storage_offset、输入 seed、计时范围。
- `correctness.json`：逐 head 最差误差、失败实例、fill/epoch/slot、非有限值、非目标字节校验。
- `metrics.jsonl`：逐次或逐窗口 raw 样本，kernel/device 区间、host wall、实际调用与 due 数；不得只留下均值。
- 代表性 profiler trace、stdout/stderr、结果摘要和构建说明。性能主结论不使用带重型 profiler 的运行。
- 每行结果标识 `measured / component_estimate / target`；后两类不能写进实测收益列。

### 测量规则

1. 编译、输入生成、初始化、绑定预热和用于独立采样的 reset 放在计时外。完整窗口内禁止每步 zero_/reset；生产必须发生的操作全部计入。
2. 单芯片无竞争，顺序配对 A/B、B/A。初筛 3 轮；最终门 5 个独立配对轮次。报告中位数、逐轮值和波动，噪声覆盖收益则为未定。
3. 同步放在测量组边界。事件区间、host 提交、kernel 执行和完整 wall 分开报告；重叠区间不能简单相加。
4. 输入来自随机/结构化契约数据及真实投影 replay。性能输入应在有效域内，不使用固定 kn=1、qn=0.5 来代替真实完整 step。
5. 冷缓存使用两个独立协议：单层 8 个 slot 集轮换；或者 24 个不同地址的层 pool 轮换。不要分配“24 pool×8 集”的全组合。
6. 8 集协议按逻辑 step 交替访问各集，连续 16 轮使每集真实推进 16 步，组总时间除以集数；24 层协议按真实层次访问，连续推进各层窗口。不得为了轮换而每次重置 fill。
7. 记录工作集和缓存行为；名字叫 cold 不构成证据。所有候选使用相同协议。
8. 对微核、完整 core 窗口和完整 projection 后 pipeline 分开定义范围；融合了 norm/gating/GQA 的候选，与 dense 相同工作范围比较。

### 数值与状态契约

- Step：逐 head 的 output、k_row、u_row 满足作者判据：relative L2≤3e-3 且最大绝对误差≤参考 head 最大值的 1%，或全部元素在一个 BF16 ULP 内。全零 head 必须严格全零；w/p 相对误差≤1e-5；fill、槽地址和目标写入符合生命周期；全部输出有限。
- Flush：用 FP64 重建“同一量化 checkpoint＋同一窗口”的当前 S。逐 head 比较解码新 checkpoint 的 mean absolute reconstruction error：参考误差≥1e-4·mean|S|时，候选≤1.01×参考；否则候选≤1e-3·mean|S|。codes∈[-127,127]，全部有限。
- 作者 `definition.json` 是数学规范；用修正后的真递推另验连续状态。不能复用已作废的 oracle ratio。
- 正确性包括全零、极小、大幅值、rank1/rank2、hot row、强抵消、窗口边界、第二窗口、非连续 idx、slot0、slot 回收。
- 非目标 slot 及 conv/FA/邻接区的原始字节必须不变。
- 量化轨迹对 dense 的长期偏离另行分析，不将单步误差门直接套到独立 dense 长轨迹；不能用不同合理舍入顺序下 codes bit-exact 替代重构误差。

## 4. R0：工件、可服务 batch 和已有显存账审计

建议入口：`scripts/audit_migration_v2.py`。优先读已有日志；只补缺失的最小采样。

### R0a：确认实际执行的是哪个算子

固化容器补丁、探针、wrapper、harness、runtime 依赖和原始日志。核对 `ascend910_93` 的实际编译头、默认头/arch35 选择、shared library 路径与加载后的 hash。对象 mtime 仅作为构建线索，不能替代加载库身份。

隔离实验分支，生产开关保持原值。为每个候选显式返回路径 ID；fallback 不得伪装成候选计时。

### R0b：先确定引擎跑得到的 batch

在相同权重、内存预算、prefix 设置和合法 max_model_len 下，复用启动/请求账确认：

- 4K 与 8K tokenized prompt、256 输出时，可持续驻留的实际 B。
- 独立 prompt 与共享 prefix 分开记账；请求总数 N 不当作实际 batch B。
- 选一个可部署主目标 B（优先从 64/96/128 中选）及 B=16 小批次检查点。
- B=256 保留为已有算子预算诊断点；只有引擎确实能维持它，才作为端到端目标。
- 现有记录不足时只补最小边界检查，不重做全并发 ramp。不可驻留点标 N/A，不靠换短 prompt 伪造同负载比较。

默认 core 输入为已规范化 q/k、g_log、beta；新融合版本另报原始 mixed QKV 的完整 pipeline 对比。

### R0c：只审计已有显存数据

复用已有 `exp1_mem_split_{A,B}.json` 与脚本，如果这些文件在运行机上。本仓库没有它们。缺少原始文件则明确缺口。不要用同一次共享前缀负载重复跑来证明普遍无收益，也不要把缺口写成“快照已经很少”。

按真实 storage/data_ptr、cache group、block ID 和 refcount 去重；区分固定预分配、活状态、冷快照、active window、FA、conv 和 padding：

- 48 个 view 的 numel 合计不是 unique HBM。
- 8 个 storage 的 allocated 账与账本 block 的物理映射必须解释清楚，不能漏乘/盲乘。
- cached map 条目不是唯一冷 checkpoint 块数；active-ref 块不自动计为可腾出的冷快照。
- 49,152 B 是 conv 尾部，不能当作全部 GDN 份额。
- 启动 token budget、逻辑在用字节、已分配 HBM、可回收空间分别列出。

产物：一份来源 manifest、可部署 B 列表、一张去重后的内存账。三者未明确，先不扩大实验。

## 5. R1：最小可靠性门——零输入、读回和多 pool

建议入口：`tests/test_migration_state_visibility.py`。

目的：堵住已有 0.025 残留、NaN 和多 pool 状态冻结对新计时的污染；不展开旧 v2 的无限定位。

### 最小矩阵

| 层级 | 输入/调用 | 检查 |
|---|---|---|
| A | 全零、随机、极小、异常幅值、强抵消；正确 64 行 add-fold 参数 | 归约输出与全部写入范围 |
| B | 真实 int8→FP16→FP32 或 hi/lo 路径；重跑 Gate 3 所需读回 | 转换完整值域、零值、有限性；禁止用未读回的快路径 |
| C | 2 个同 shape 异址 pool，A/B/A；稳定与新 idx 地址 | 绑定、executor cache、stream、目标 buffer |
| D | 24 pool、连续 96 步、非连续 slot、回收再初始化 | 6 窗状态、生命周期与邻接字节 |

每次同步后的证据链：整块 backing storage D2H 按字节解析、小型设备 readback、Torch typed view。只有三条链定位结果后才归因；不能用“fallback=0”证明状态正确。

检查产生/消费 buffer 的事件、V/S 与 MTE 顺序、初始化覆盖和对齐。必要的同步是正确性成本，留在候选路径中。

**通过：** 所有应写位置有确定值，零 head 严格为零，读回一致，非目标字节不变。首轮只要求 A–C；完整候选进入 R5 前必须通过 D。失败候选停止计时和融合。不得关闭检查后继续报速度。

## 6. R2：两个 codes 读出的硬件可行性筛选

建议入口：`scripts/bench_codes_readout.py`。

输入来自真实 composite pool：int8 codes、FP32 scales、真实 key/query。输出是两个 FP32 读出。计时包含实际读取、精确转换、scaled query/key 准备、矩阵计算、结果合并、GM 临时量、绑定与调用。预先转换的矩阵不能悄悄成为免费输入。

| 路线 | 实现 | 要回答的问题 |
|---|---|---|
| V | 正确 add-fold；批量转换；同一 codes tile 服务两个读出 | Vector 在实际数据与搬运下还有多少空间？ |
| C | 独立、已可运行的 FP16 矩阵算子；codes 精确转 FP16；k/q hi+lo 打包 | Cube 的 padding、launch 与转换之后是否仍更便宜？ |
| H | Cube 做 codes，小规模向量操作做缩放/合并 | 两个算子/GM 中间量的总成本是否可接受？ |

优先 B=16、可部署目标 B、B=256。明确每 head 的独立矩阵、layout、M/N/K padding、累加/输出精度与临时 HBM。不能把不同 head 的状态当作相同权重共享。

本阶段不要求单核 MIX。当前 MIX 接线失败不应阻塞独立矩阵算子的正确性/成本验证；多调用成本必须计入。只有它有可用收益，才考虑合并发射。

**数值门：** 与 FP32/FP64 codes-only 参考逐 head 对拍，覆盖 scale 范围与强抵消。直接把完整 decoded S 舍入 FP16/BF16 不属于这条路线。

**预算门：** 这是 step 的一部分。结果必须给剩余预算，不能仅报相对旧逐行实现的倍数。若仅这部分就达到 R0 对应的完整 step 免费-flush 上限，当前组织停止。若超过目标但未触及必要条件，仅允许一轮有 profiler 根据的修正。

单核 MIX、persistent 调度、转换技巧都不是先验通过条件；有正确实测和总成本空间才是。

## 7. R3：完整直接读出 step

建议入口：`tests/test_direct_step_v2_contract.py`、`scripts/bench_direct_step_v2.py`。这是新实验候选，不等于旧 step v2 接线。

沿用当前合法 composite layout、L=16/R=4、idx/slot 生命周期。对 x=kn 和缩放 qn：

`Sx = pn * [sv ⊙ (codes @ ((sk⊙x)/127)) + Σ_r q_r*(u_rᵀx)] + Σ_live_j (w_j*at)*ubuf_j*(kbuf_jᵀx)`

再计算 `u_new=beta*(v-Skn)`、`o=Sqn+(kn·qn)*u_new`，原地追加更新、衰减 w/p、写 fill。

### 实施顺序

1. 分别激活 codes、compensator、window、当前更新，确认数学和 decay。
2. 接入 R2 通过的读出。Vector 路径必须批量处理 live window/低秩点积，避免重新展开完整外积矩阵。
3. 使用 h→h//2 直接访问 16 个 key heads；不在 host repeat_interleave。
4. 先保留两边同范围的已规范化输入；正确后增加 norm/gating 融合版，比较完整 projection 后 pipeline。
5. 如用 FP16 checkpoint shadow 或跨窗口常驻转换缓存，另列其转换成本与额外物理内存；不能仍声称原 INT8 流量/容量已复现。

覆盖 fill=0/1/8/15/16、slot0、非连续 idx、两窗、96 步和 24 pool。读出 error、写回 k/u/w/p/fill 同时检查。死 buffer 值可能不有限时必须按生命周期处理，不能靠 0*NaN 消除。

性能报告 fill=0/8/15 与真实连续窗口均值；包括实际 load/cast、全部补偿/窗口工作和写回。代码不构造 S，也不等于天然快。

**通过：** 完整 step 契约全绿；在可部署目标 B 和 B=256 分别给出真实剩余 F 预算。只比旧 step 快≥10%不构成通过。超过免费-flush 上限的当前组织停止。

## 8. R4：flush 的新数学/硬件组织

建议入口：`tests/test_flush_gpu_contract_port.py`、`scripts/bench_flush_gpu_contract_port.py`。

固定 checkpoint 格式、q0、一次子空间迭代、R=4、round-to-nearest。不能通过 warm-start、减迭代、少做 smoothing 或宽松精度换速度。

### R4a：复现 GPU 的残差/旧低秩分解

按 NPU 的 S[V,K] 方向，可写：

`S = B + A`
`B = p*dequant(old_residual) + Ubufᵀ*diag(w)*Kbuf`
`A = p*Q_oldᵀ*U_old`

先 FP32 reference 验证代数；再试将 B/残差保留为 FP16、旧 A 保持因子形式并以高精度代数修正参加拟合和残差计算。不要先把完整含大补偿项的 S 一次转 FP16，再减掉它。

先用正确的现成矩阵算子和简单 epilogue 跑通；GM 中间量/转换/调用全部计时。是否放入 UB、是否合并 AIC/AIV，属于后续成本优化。

### R4b：矩阵与正交化

- 窗口重建、子空间乘积、新旧低秩残差优先形成批量矩阵计算。
- 小 Gram/CholeskyQR2 使用明确精度和 relative pivot 保护，验证 rank-deficient、零和极小 head；数值通过后才替换现有正交化。
- smoothing 的列求和/行最大值比较批量 Vector 与矩阵组织，选择全 flush 更快的一种；不将某个归约的倍数推广至整个 fold。
- 控制临时状态、padding、转换与 producer/consumer 依赖；每次只改变一个可归因因素。

旧串行 pass 链继续保持停线。若三段不可消除成本已超过总预算，不再围绕另一段做局部乐观拼装。

### R4c：设备 due 和负载均衡

all-due 数值及成本具备希望后才优化调度。

以设备 fill 为真值；优先 scan/compact→均匀 worklist→compute 的可验证路径。将同一请求的 32 head 分散到可用核，不让集中 due 全部压在一个 block。独立两核的 dispatch/GM 队列成本计入。

若各 CTA 同时扫描并计算，不能在别的 CTA 扫完前改变其 due 标志；可使用快照队列或作者式 fill=L 标记，由下一次 step 打开新窗。该约定必须连同 w/p 重置、active idx 和 slot 回收一起验证。

due 矩阵：0、一个 head（仅单测）、一个完整请求、约1/16请求、1/2请求、全部；均匀与集中分布。生产主结论使用请求级相关 fill，不只测独立随机 head。

**通过：** 每 head reconstruction 门及第二窗口全绿；非 due 原始字节不变；全部实际 flush 调用成本落在 R3 剩余预算内。仅 nothing-due 改善不解锁当前 all-due 预算失败的实现。

## 9. R5：完整窗口与多层配对决策门

建议入口：`scripts/bench_window_migration_v2.py`。

版本：

- D：实际生产 FP32 dense op。
- M：当前生产 step＋flush v1，作为差距定位。
- V：通过的直接 Vector step＋通过的 flush。
- C/H：通过的 Cube/混合直接 step＋通过的 flush。

旧 v2 不默认加入。不要使用 eager materializing Torch 作为速度主对照。

### 最小测试矩阵

| 轴 | 首轮决策点 | 通过后扩展 |
|---|---|---|
| B | 16、可部署主目标、256 | 64/96/128其余点；512仅在可运行时 |
| 生命周期 | 连续16步、至少6窗口；随后256步 | 加入/结束、回收、缓存恢复 |
| pool/cache | 单层8集；24层各自不同地址 | 高slot、非连续idx |
| due | lockstep all-due；合法历史生成的 staggered fill | 集中/均匀请求边界 |
| scope | core完整窗口 | 相同范围 norm/gating/GQA pipeline |
| capture | eager | 双方支持的相同 graph 模式 |

staggered 初态通过真实参考历史生成，不凭空改 fill 后声称数学连续。

**进入 R6 的门：**

1. 数值、24 pool、生命周期全通过，无隐式 fallback。
2. 可部署主目标 B 的完整窗口中位数速度比≥1.10，5轮方向一致且波动不覆盖门。
3. B=256 按原对齐协议达到≤14.527 ms，才写该点通过；不可部署的 B=256 收益只作为算子结果。
4. 若只有 B=128 或其他实际可部署点通过，可进入受限 batch 的集成验证，但不得写 B=256 或普遍复现成功。
5. 必要调用、临时 HBM、转换和写回已经计入；组件和 probe 相加不能过门。

1.00–1.10 仅记弱收益，默认不扩展引擎/分配器工程。持续<1停止当前组织。仅有更快 primitive、没有完整正确窗口，结论仍为 NO-GO。

如果新的合法 dense 优化更快，使用它重新收紧门；当前默认不改 dense。论文 GPU 的绝对时间和组件拼装不作为 NPU 上限/下限。

## 10. R6：引擎保留收益，而非重新扫描所有负载

建议入口：`scripts/bench_engine_migration_v2.py`。依赖 R5。

先最小接线，只启用已通过 batch/shape 的候选。状态表示不能在每步 dense/quant 间任意切换；任何转换或边界切换均有单独正确性与成本记录。

### 三个测量层级

1. 固定实际 B、4K context、256固定输出的纯 decode。性能测试可 ignore_eos，所有模式一致；先完成 prefill，再开始 decode 计时。
2. 相同参数的完整 prefill＋decode，记录 TTFT和所有转换成本。
3. 最后只补 8K 与实际服务形态；不以请求 N 代替 B，也不通过不同输出长度补容量点。

先 eager，再双方均支持的相同 PIECEWISE 等模式。FULL 当前 Ascend GDN 不支持，不作为默认任务；不能一方 graph、一方 eager 对比。

记录实际每步 B、step/flush/count/due、host 提交、NPU trace、内存、fallback、输出实际 token、finish_reason。吞吐分母与墙钟区间匹配。3.295 s 等历史残差只有直接测量后才归为 host/launch。

### Amdahl 筛选

用实际目标 B 的 dense 时间拆分估算收益空间。B=2 的历史4.5%不能用于高 B，也不能承诺可观速度。若候选算子加速 s、可优化份额 f，则预测纯decode上限为 `1/(1-f+f/s)`，与实测对照。

**通过：** 可部署目标 B 的纯 decode 稳健≥5%；完整负载的转换成本没有抵消收益；零fallback、正确状态。这是计划提出的工程门，不是论文保证。

若收益消失，最多一轮 trace 驱动诊断 norm/GQA、临时分配、描述符、图边界和提交；没有恢复则停止集成扩展。小 B 保持 dense 路径是否可行，要按请求生命周期验证，不能临时逐步切换格式。

## 11. R7：精度贯穿算子；最终任务评测按候选收敛

### R7a：teacher-forced轨迹

R3/R4 正确后，用相同 token 序列比较 dense FP32、作者数学 LQ oracle、当前 mode3 和候选：

- 先1K，再8K；通过后测试真实可执行的更长长度。
- 每窗记录最差head重构误差、层输出、logits/NLL、非有限值；定位平台实现误差与算法量化误差。
- 用单层真实QKV replay可做64K状态漂移诊断，但不能称为64K整模型推理复现。
- 不通过重置状态、截断或改FA上下文来假装等价长推理。

### R7b：任务筛选与正式非劣

候选收敛后先固定筛选集，覆盖长推理和代码；50–200题规模只用来发现明显回归，不证明1个百分点非劣。

最终按论文模型卡/Appendix E 的 prompt、thinking、采样、chat template、输出上限与3 seeds，对AIME、GPQA、LiveCodeBench、MMLU-Pro、GSM8K作配对评测。正式任务允许自然EOS；性能固定输出的ignore_eos不进入准确率测量。

任务非劣界暂定每任务下降不超过1个百分点，报告配对置信区间和seed波动；下界不能支持非劣时写未定。输出长度、截断比例、失败/超时同时报告。论文长输出配置若未真实支持，列出未覆盖项，不能宣称完整精度复现。

若量化oracle本身与dense有差距，单列算法/评测差异；若候选比oracle明显差，回到数值定位。逐token分叉不是任务失效的充分证据，少量token一致也不是任务非劣的证据。

## 12. R8：容量与固定内存服务——独立条件门

容量实现默认暂停。R0c 的账可以先做；完整 allocator 改造默认等待 R5/R6 和空间价值同时成立。

### R8a：纸面价值门

以可回收冷 GDN checkpoint 的真实字节 C 为基数，理想压缩上界近似：

`saving_max = C*(1-19456/65536)`

再减 active window、alignment、索引及恢复临时量。这不是把整个31.49 GiB池都替换，也不是用conv比例估算GDN。

暂定潜在可用空间至少达总HBM的15%，或能实测跨过有价值的接纳边界，才投入独立分配器工程。15%是工程筛选阈值；若目标另为少量显存节省，应单独定义项目价值。

仓库里还没有一份过线的冷快照账。只有审计和静态估算显示“独立长前缀、少量活跃请求”可能过门，才新增一组容量定向负载。独立前缀不是预设成功场景，还需要计入FA KV、prefill和驻留账。

### R8b：分配与恢复

满足价值门后，先独立checkpoint/active-window生命周期，明确持久槽与活跃槽的映射；FA、conv、checkpoint cadence不因slot缩小而被意外改变。保存非窗口边界缓存必须显式折叠或同时保存窗口，不能丢更新。

覆盖chunk边界、cache hit/miss、hot/cold restore、slot回收、高slot、跨组alias、conv/FA哨兵、图捕获和池迁移。分配账必须加回真实 unique storage。旧compact page覆盖V的路径不复用。

### R8c：固定内存服务

同一权重、HBM预算、cache/graph设置、max_model_len，比较dense与通过候选。N只取dense容量以下、两个实测接纳边界之间、量化边界以上三个点。独立/共享prefix、32/256输出分开。

指标：实际resident/running/waiting、block占用、抢占/重算、TTFT、TPOT、完成率、实际output token、总wall与吞吐。服务SLO需预先固定；离线吞吐不替代SLO。

容量与速度按 `B_lq/T_lq(B_lq)` 对 `B_dense/T_dense(B_dense)` 分析，完整服务以实测采信；不将已有0.54×乘1.145/1.26。

只省显存、只减排队或只增加接纳，分别宣称对应结果。若所有完整服务更慢，继续停止吞吐收益宣称。

“dense活状态＋压缩历史checkpoint”的memory-only路线可以作为独立项目，但本轮默认不启动；它仍必须先过R8a并验证恢复精度，不能借用活量化的算子倍数。

## 13. 分轮执行包与退出决定

| 轮次 | 执行范围 | 每轮必须交付 | 退出决定 |
|---|---|---|---|
| 第一轮 | R0；R1 A–C；R2三点小矩阵 | 正确来源、可部署B、现有内存账、正确读出与总成本 | 没有可靠/有预算空间的路线，停在这里 |
| 第二轮 | R3完整step；R4分解与矩阵flush；R1 D | 逐head契约、实际全成本、剩余预算、24pool状态 | 两项不能组成可行窗口，停止当前组织 |
| 第三轮 | R5真实窗口 | 5轮配对、cold/24pool、lockstep/staggered、原始日志 | 仅R5通过才开R6 |
| 第四轮 | R6最小引擎；R7轨迹/筛选 | 实际B的decode与完整wall、任务回归筛选 | 收益被集成抵消且一轮诊断无恢复则停 |
| 条件轮 | R7全量；R8容量/服务 | 任务非劣统计、物理容量与服务结果 | 四类收益各自PASS/FAIL/未定 |

资源限制：每条算子组织首轮最多两次实现迭代；额外一轮必须由明确profile/正确性证据支持。每次只变一个因素，先三点筛选再扩矩阵。性能同芯片串行采信；独立正确性可以另机执行。

## 14. 最终结论和可审阅交付

最终分开报告：

1. 单层完整窗口：相对生产dense，在哪些真实B通过。
2. 固定B纯decode与完整E2E：是否保留算子收益，适用context和图模式。
3. 固定内存服务：物理空间、实际接纳与吞吐/SLO是否改善。
4. 任务精度：数值契约、长轨迹、各任务非劣是否成立。

每项一页：假设→版本/输入→实际计时范围→最差head/状态正确性→配对结果→profile解释→可支持的结论→未覆盖项。

只记录算过且正确的结果。完整窗口未过门，不通过局部快核、理论拼装、弱dense、启动预算或少量token一致替代。
