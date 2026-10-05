# 待定位 bug：gdn_lq_flush ILP 重构后 u 写回损坏（WIP 提交 2ec1a0d）

## 症状（tests/test_flush_ascendc.py，当前 .so 18:50 构建）

逐字段对比 torch 参考（96 步、每窗口末 decode_state）：

| 字段 | 状态 |
|---|---|
| q (qv, MGS 正交因子) | **逐位一致** |
| s_k / s_v（来自残差 E 的平滑 scale） | rel 1e-5 / 1e-4 —— 一致 |
| codes | 2.5% 元素 tie 翻转（正常） |
| w / p / fill | 一致 |
| **u (uk = Sᵀqv)** | **rel diff 31.6%** |
| checkpoint 整体 | rel ~14% |

## 推理

E = S − qv·ukᵀ 是正确的（s_k/s_v 由 E 导出且吻合），说明 kernel 内参与
E 计算的 uk 值正确、MGS 全对。**坏的只有 uk 的 GM 写回路径**。

## 主要嫌疑

重构把写回改成 UB 暂存区批量拷贝：
`DataCopy(gmU[0], dh[F_R*F_V*17], F_R*F_V)`（op_kernel/gdn_lq_flush.cpp）
- `dh[F_R*F_V*17]` 的偏移 17 是否算错（staging 布局 dh[] 的槽位分配）
- uk 在 pass 3 后是否被后续 pass 覆盖/未刷新进 dh 对应槽位
- 对照：gmQ 写回（qv, 正确）用的是哪个偏移/路径，差异即答案

## 复现

```bash
cd /mnt/workspace/leapquant-npu
export LD_LIBRARY_PATH=/home/developer/Ascend/nnal/atb/9.1.0/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH
python3 tests/test_flush_ascendc.py   # 断言失败，打印逐字段 diff
```

对照基线：重构前（vllm-ascend commit 6bcab21 的 kernel）全部字段 PASS。
重构 diff：`git diff 6bcab21 2ec1a0d -- csrc/attention/gdn_lq_flush/op_kernel/gdn_lq_flush.cpp`
