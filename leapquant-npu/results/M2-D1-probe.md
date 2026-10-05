# M2-D1 探针结果：LeapQuant 核心 GEMV 上 cube（fp16 dequant → cube GEMV）

日期：2026-10-05 ｜ 机器：2×Ascend910（64GB HBM，npu-smi 显示 "Ascend910"），卡 5 ｜ CANN 9.1.0 ｜ soc 分支 ascend910_93（__NPU_ARCH__=3003，dav_l300）

## 三行 go/no-go

1. **编译：GO。** 探针 op `GdnLqGemvProbe`（AIC-only，fp16 16×128×128 GEMM/实例，fp16×fp16→fp32）在 ascend910_93 构建通过并安装（kernel 元数据 `coreType: MIX` 路径弃用后改为 AIC-only；`libmatmul_intf.h` + host 侧 `matmul_tiling::MatmulApiTiling::GetTiling` 生成 TCubeTiling，`baseM=16 baseN=128 baseK=128 singleCore=16×128×128` 单 tile）。int8×fp16 未尝试（任务书注明"不用试"；910 cube 仅支持 fp16×fp16→fp32 / int8×int8→int32 是既有事实）。
2. **数值对拍：GO。** 对 torch fp32 参考 `o[g,n,v]=Σ_k(codes[g,k,v]·scale[g,k])·q[g,n,k]`，全部 9 个配置（G=512/2048/8192 × N=1/2/8）rel L2 ≈ **2.1e-4**，远低于 3e-3 门槛。
3. **计时：GO。** 纯 cube kernel：G=512 → 0.036ms，G=2048 → 0.06–0.08ms，G=8192 → ~0.41ms（与 triton step 同口径重测值 2.097 / 3.018 / 6.031ms 对照，见下表）。

## 计时表（纯 kernel ms，warmup→sync→perf_counter→sync，照 bench_step_ascendc.py 口径）

| G (实例数) | N=1 | N=2 | N=8 | triton step 同口径重测 | 旧 triton 数字（仅供参考） |
|---|---|---|---|---|---|
| 512  (B=16)  | 0.036–0.037 | 0.036–0.037 | 0.036–0.037 | 2.097 ms | 0.444 ms |
| 2048 (B=64)  | 0.070       | 0.059–0.076 | 0.067–0.076 | 3.018 ms | （无） |
| 8192 (B=256) | 0.412–0.418 | 0.409–0.412 | 0.410–0.412 | 6.031 ms | 4.238 ms |

- 计时仅含 cube GEMV op 本身（A/B tile 由 torch 在计时循环外备好）；N=1/2/8 无差异符合预期（M 恒为 16，pad 行是零乘）。
- **口径警示**：本机同口径重测的 triton step 与任务书给的两个旧数字（0.444/4.238ms）都对不上（B=16 差 4.7×，B=256 差 1.4×），报告里的对照请以重测值为准，旧数字可能是别的口径/别的机器状态。
- 换算到 step v2 语境：cube GEMV 在 G=8192 时 0.41ms，占当前 triton step（6.03ms）的 ~7%；即使叠加上 AIV dequant staging 的开销，cube 路线相对 vector reduce 瓶颈（B=256 时 4.2–6ms/层量级）有明确余量。

## 对拍误差（rel L2，torch fp32 参考，seed 0）

| G | N=1 | N=2 | N=8 |
|---|---|---|---|
| 512  | 2.08e-4 | 2.06e-4 | 2.06e-4 |
| 2048 | 2.07e-4 | 2.07e-4 | 2.08e-4 |
| 8192 | 2.08e-4 | 2.07e-4 | 2.08e-4 |

误差来源：A 侧单次 fp32→fp16 舍入（scale·q 在 fp32 乘完再转 fp16），B 侧 int8→fp16 精确，cube fp32 累加。恒为 ~2e-4 且与 G/N 无关，符合该误差模型。

## 数据通路结论

- **dequant 后 fp16 tile 只能经 GM staging 喂 cube**：910（dav_l300）vector 核不能写 L1；cube 的 `SetTensorA/B` 从 GM（本次实现）或 L1 读。本次探针里 cube 直接读 torch 备好的 GM tensor（`IterateAll` 写回 GM），跨 kernel 顺序由 stream 保证。
- **AIV 片上 dequant→GM staging 路径已验证**（原 MIX 方案中的 Stage 代码，AIV-only 下实测）：codes int8→fp16 逐行 Cast 后 GM staging 读回**逐位一致**（例：`[67,-73,106,8,96,77,-60,36]`）；A tile = fp32(q)·scale 再转 fp16，读回与 torch 参考误差 ~1e-4 量级。即"vector 算完写 GM staging（L2 命中）、cube MTE2 拉取"的通路本身成立。
- L2 命中证据：无直接 counter 测量；依据是通路结构（GM staging 在 44KB×G 内被同 launch 内写后读，G=8192 时 staging 全集 360MB 远超 L2，单次 staging 访问模式为"写一次→cube 读一次→丢弃"，带宽账已在计时内）。

## 重大问题：MIX（AIC+AIV 同 kernel）在本机不可用（no-go for fused path）

任务原设计是 MIX kernel（AIC:AIV=1:2，AIV dequant + 硬件 flag 握手喂 cube）。实测本环境 **MIX 启动路径损坏**：

- `KERNEL_TYPE_MIX_AIC_1_2`（KERNEL_TASK_TYPE_DEFAULT 声明，元数据正确生成 `kernelType: MIX_AIC, taskRation: 1:2, crossCoreSync: 1`）经 aclnn 启动时走 **fftsplus task**，**空 kernel（双端立即 return）也必然故障**：
  `exception of fftsplus aicore error, core id 0, error code = 0x800000 ... MTE accesses an invalid GM address ... GdnLqGemvProbe_..._mix_aic+0xb4`。
- 换 blockDim（AIC 数 20 / AIV 数 40）、加 `KERNEL_TASK_TYPE(0, ...)`、kernel 体内容均无效——与 chunk_kda_fwd/gmm 的元数据逐项比对无差异，判定为**该设备/驱动组合上 MIX fftsplus 调度路径本身的问题**（npu-smi 驱动 25.5.5，CANN 9.1.0）。
- 另外 dav_l300 头文件里 `CrossCoreSetFlag/NotifyEventImpl` 是**空函数**、`CrossCoreWaitFlag<0x2>` 只映射为本地 `PipeBarrier`——即使 MIX 能启动，任务书设想的"AIV 写 flag / AIC 轮询"硬件握手在此 arch 也不存在，只能做 GM 内存 flag 轮询（`dispatch_ffn_combine_w4_a8` 在 ascend910_93 列表里，但其 CrossCore flag 调用在本 arch 同为空操作，该 op 在同环境能否真正跑通未验证）。
- 对照：**AIV-only 与 AIC-only kernel 都能正常启动运行**（空壳测试均过）。

**对 step v2 的含义**：单 kernel 融合（dequant→GEMV 同 launch）在本机不可行；可行走法是 **stream 序的三段式**（AIV dequant staging op → AIC GEMV op（本探针）→ AIV epilogue op，或首尾用 torch 算子替代），或者先在驱动/CANN 升级的环境上重试 MIX。本次探针证明的 cube GEMV 本体（编译/数值/性能）不受影响。

## 其它意外与坑（按踩坑顺序）

1. **aclnn 161001（SoC verification failed）**：binding 没把 `_cann_ops_custom` 加进 `ASCEND_CUSTOM_OPP_PATH`（现有测试靠 import vllm 触发 `bootstrap_custom_op_env`）。已在 `gemv_probe.py` import 时自行注册 vendor 路径。
2. **AIV-only 下 CopyOut（C 读回→o 写出）不稳定**：同一代码从"多 vec 核定点故障（vec error 0x130000000c，PC+0x6dc）"到"整体挂死"随小改动切换，Level-4（仅 Stage）始终稳定。未定位到单条指令级根因（疑似事件/控制流相关的编译器问题）；因 MIX 已不可用、AIC-only 方案不需要这段代码，**未继续追**。若将来做三段式的 epilogue op，这段逻辑要重写并重新趟（建议先用 FetchEventID 替代固定事件 id、串行循环替代 lookahead）。
3. 任务书里的 kernel 侧已知坑（WholeReduceSum repeat 形态、运行时索引局部数组、大 count Cast、小尺寸 DataCopyPad）在本探针代码中全部规避（未用 reduce；Cast 全部 128 元素行；无小尺寸裸 DataCopy）。
4. autogen 陷阱实测成立：改 def/infershape 后必须删 `build/autogen/aclnn_gdn_lq_gemv_probe.*`、`gdn_lq_gemv_probe_proto.*` 及 `build/binary/ascend910_93/bin|cache_actions/gdn_lq_gemv_probe` 里旧 hash 产物，否则 publish 报 "published artifact changed outside its owning action"。

## 代码与复现

- op：`/mnt/workspace/vllm-ascend/csrc/attention/gdn_lq_gemv_probe/`（op_host def/infershape/tiling + AIC-only kernel；已注册进 `csrc/build_aclnn.sh` 的 ascend910_93 列表）
- binding：`/mnt/workspace/leapquant-npu/gemv_probe.py`
- bench/对拍：`/mnt/workspace/leapquant-npu/tests/bench_gemv_probe.py`
- 构建：`cd /mnt/workspace/vllm-ascend/csrc && bash build_aclnn.sh /mnt/workspace/vllm-ascend ascend910_93`
- 运行：`LD_LIBRARY_PATH=/home/developer/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH ASCEND_VISIBLE_DEVICES=5 python tests/bench_gemv_probe.py`
- 基线回归：test_step_triton / test_step_ascendc / test_flush_ascendc / test_flush_strided / test_pools_parity 在探针完成后全部复跑通过。
