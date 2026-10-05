# hybird-quant: LeapQuant on Ascend NPU (vllm-ascend + Qwen3.5-9B GDN)

将 [LeapQuant](https://arxiv.org/abs/2609.38166)（线性注意力循环状态 per-window INT8 量化，
契约源自 [NVlabs/kda PR #10](https://github.com/NVlabs/kda/pull/10)）移植到
**vLLM v0.30 (vllm-v030 fork) + vllm-ascend + Qwen3.5-9B（GDN 混合架构）+ Ascend 910**
组合的适配工程。

## 原理（GDN 变体）

线性注意力层每 decode token 需读写 128×128 fp32 状态（2MB/头），带宽瓶颈。
LeapQuant 把状态改为复合格式**原地**存放在原 ssm_state slot 内（零额外显存）：

- int8 codes（平滑残差，per-column s_k × per-row s_v 缩放）
- 4 个 fp16 Compensator Tokens（rank-4 高精度补偿，走真实 token 的 rank-one 更新路径）
- 16 步 bf16 rank-one 更新缓冲（w/p/fill 计数器）

窗口内 decode 从"固定 checkpoint + 缓冲更新"重建输出（不物化状态）；窗口边界 flush 一次：
重建 → 幂迭代/CholeskyQR 拟合新补偿 token → 残差平滑 → INT8 量化。

## 仓库结构

```
leapquant-npu/            # 独立验证/基准 harness（不依赖 vLLM 启动即可跑的部分）
  ref/                    # PR #10 的 definition.json（精确数学契约，Apache-2.0）
  scripts/
    smoke_qwen35.py       # 全模型离线生成冒烟（--out 导出 token_ids）
    measure_drift.py      # prompt-logprob drift 测量（fp32/leapquant/bf16 三模式）
    compare_gen.py        # 两份生成 dump 的逐 token 对比
  tests/
    test_pools_parity.py  # pools vs 稠密 fp32 oracle 单层对拍（NPU）
    test_step_triton.py   # triton step kernel vs torch pools 对拍
    test_flush_triton.py  # triton flush kernel vs torch pools 对拍（kernel 未完成，见下）
    replay_dump.py        # 真实权重 dump 回放对拍（torch oracle vs fla_npu 稠密 kernel）
    bench_step.py         # step kernel 微基准
vllm-ascend/vllm_ascend/  # vllm-ascend 侧改动的文件镜像（按仓库相对路径）
patches/vllm-ascend-leapquant.patch  # 上述改动的完整 git diff（相对本 workspace 的 vllm-ascend 研究分支）
results/                  # 实测结果摘要
```

## vllm-ascend 改动清单（5 个文件）

| 文件 | 改动 |
|---|---|
| `vllm_ascend/envs.py` | 新增 `VLLM_ASCEND_GDN_LEAPQUANT`（0=关/1=torch/2=triton step + torch flush） |
| `vllm_ascend/ops/gdn.py` | `_forward_core` decode/mixed 分支接入 LeapQuant；prefill 后 resync；spec/chunked-prefill 守卫 |
| `vllm_ascend/ops/gdn_leapquant.py`（新） | 复合状态 pools（原地视图）+ torch 版 decode_step/flush/init_from_dense |
| `vllm_ascend/ops/triton/gdn_leapquant_step.py`（新） | Triton-Ascend step kernel |
| `vllm_ascend/ops/triton/gdn_leapquant_flush.py`（新） | Triton-Ascend flush kernel（**实验性，BiSheng 编译挂起，未启用**） |

关键设计：**原地存储**。vllm 的 32GB KV 池预算已用尽，独立分配状态池必然 OOM；
复合格式（~0.9MB）装进每 slot 固有的 2MB fp32 状态区，通过 as_strided 类型化视图访问。

## 使用

```bash
# 环境：Ascend 910，CANN 9.1.0，torch_npu 2.10.0.post4，triton-ascend 3.2.2
# 注意：需 ATB 库路径，否则 torch_npu ATB 插件加载失败
export LD_LIBRARY_PATH=/home/developer/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH

# 基线 / LeapQuant(torch) / LeapQuant(triton step)
python3 leapquant-npu/scripts/smoke_qwen35.py --max-tokens 48 --out /tmp/base.jsonl
VLLM_ASCEND_GDN_LEAPQUANT=1 python3 leapquant-npu/scripts/smoke_qwen35.py --max-tokens 48 --out /tmp/lq.jsonl
VLLM_ASCEND_GDN_LEAPQUANT=2 python3 leapquant-npu/scripts/smoke_qwen35.py --max-tokens 48 --out /tmp/lq2.jsonl
python3 leapquant-npu/scripts/compare_gen.py /tmp/base.jsonl /tmp/lq2.jsonl

# 单层对拍 / 微基准（需 NPU）
python3 leapquant-npu/tests/test_pools_parity.py
python3 leapquant-npu/tests/test_step_triton.py
python3 leapquant-npu/tests/bench_step.py
```

模型：`/mnt/workspace/models/Qwen3.5-9B`（32 层，24×GDN + 8×full-attention，
HK=16/HV=32/K=V=128，conv=4）。脚本内可 `--model` 指定路径。

## 验证结论（实测，Ascend 910）

- 单层对拍（合成输入）：输出 rel err ≤ 0.57%，窗口末状态重建 err ~0.5%。
- 真实权重 dump 回放：LeapQuant 逐步输出 vs fla_npu 稠密 kernel **0.05%–0.19%**（bf16 噪声级）。
- 端到端 48 token 贪心生成与 fp32 基线逐 token 一致；768 token 在 27/138 处近tie翻点。
- **3.2K token logprob drift 与一次 fp32 参考运行逐值相同**（量化误差低于运行间噪声底）。
- step kernel 微基准（B=16/64/256）：0.46/1.23/4.34 ms，**3.8–15× vs torch step**。

## 已知限制（原型阶段，均有显式报错守卫）

1. **spec decode 不支持**（dense spec 路径会把复合状态当 fp32 读）。
2. **chunked prefill / align 模式 prefix-cache 复用不支持**
   （continuation 会把复合 checkpoint 误读为 fp32；实测 mm warmup 的 chat 模板前缀即可触发）。
3. eager torch flush 是性能瓶颈（~60 次 launch/层），triton 版 flush 受 BiSheng 编译器限制，
   **下一步应直接做 AscendC 版 flush**（含 PR 契约要求的设备侧 due-slot 扫描，
   复用 sketchssm-npu 的 gdn_sketch_step op_host/op_kernel 骨架）。

## 路线图

- [x] torch 级全链路 + 精度验证
- [x] Triton-Ascend step kernel（对拍通过，3.8–15×）
- [ ] AscendC flush kernel（triton 版 UB/编译器受限，转 AscendC）
- [ ] MambaSpec 复合状态正式集成（prefix caching 原生 checkpoint → 41–56% 显存收益）
- [ ] ACL graph 化（decode/flush 图捕获，参照 gdn_attn_builder sketchssm padding 方案）
- [ ] spec decode 支持（dense 回退 + 复合状态重同步）
- [ ] 64K 级长上下文误差曲线 + AIME 类长链任务 + 端到端性能基准

## 许可

`leapquant-npu/ref/` 内的契约定义源自 NVlabs/kda PR #10（Apache-2.0）；
其余代码遵循仓库 LICENSE。
