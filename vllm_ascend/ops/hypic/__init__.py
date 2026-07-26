# SPDX-License-Identifier: Apache-2.0
# Hypic: position-independent caching (PIC) for hybrid-attention models.
#
# This package implements the model-side primitives of Hypic
# (arXiv:2607.01299) for Qwen3.5-style GDN hybrid stacks:
#
# - state.py:      the cached per-segment tuple (T_C, S_{C|0}, conv_state)
#                  and the constant-time state composition law
#                  S = T_C @ S_prefix + S_{C|0}  (Eq. 6 of the paper).
#                  Pure PyTorch, no vllm/vllm-ascend imports, CPU-testable.
# - transition.py: extraction of the segment-cumulative transition operator
#                  T_C from the chunked GDN recurrence by re-running the
#                  h-recurrence with h0 = I and the write term zeroed.
#                  Requires NPU (Triton FLA kernels).
#
# Scheduler-side PIC (segment routing, seam-window piecewise prefill,
# segment parallelism) is intentionally NOT in this package; it lives in
# the scheduler / model-runner layer and is tracked separately.
from vllm_ascend.ops.hypic.state import SegmentCache, compose_states

__all__ = ["SegmentCache", "compose_states"]
