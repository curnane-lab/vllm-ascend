# SPDX-License-Identifier: Apache-2.0
"""Hypic segment cache tuple and linear-attention state composition law.

For a linear-attention (GDN) layer with token-level recurrence

    S_i = T_i @ S_{i-1} + u_i,        S_i in R^{K x V},  T_i in R^{K x K},

a segment C is fully described, independently of any prefix, by the tuple

    (T_C, S_{C|0})

where T_C = prod_{t in C} T_t is the segment-cumulative transition operator
and S_{C|0} is the end-state obtained by unrolling C from a zero initial
state. Composing a prefix state S with a cached segment is exact and takes
time independent of the segment length (Eq. 4/6 of arXiv:2607.01299):

    S_{prefix + C} = T_C @ S + S_{C|0}.

This module is deliberately free of vllm / torch_npu imports so the
composition math can be validated on CPU.
"""

from dataclasses import dataclass

import torch


@dataclass
class SegmentCache:
    """Per-segment reusable cache for one linear-attention layer.

    Attributes:
        transition: segment-cumulative transition operator T_C,
            shape [..., K, K]. For scalar-decay families this degenerates
            to a single scalar per head; for GDN (dense family) it is a
            full K x K matrix per head.
        zero_start_state: end-state S_{C|0} of the segment unrolled from a
            zero initial state, shape [..., K, V].
        conv_state: trailing causal-conv state of the segment, if the model
            prepends a conv1d to the QKV projection (e.g. Qwen3.5). Needed
            to warm up the conv at reuse time. Layout is model-specific and
            opaque to this module.
    """

    transition: torch.Tensor
    zero_start_state: torch.Tensor
    conv_state: torch.Tensor | None = None

    def numel_bytes(self) -> int:
        total = self.transition.numel() * self.transition.element_size()
        total += self.zero_start_state.numel() * self.zero_start_state.element_size()
        if self.conv_state is not None:
            total += self.conv_state.numel() * self.conv_state.element_size()
        return total


def compose_states(
    prefix_state: torch.Tensor,
    transitions: torch.Tensor,
    zero_start_states: torch.Tensor,
) -> torch.Tensor:
    """Compose a running state from a prefix state and cached segments.

    Applies the composition law right-to-left in segment order:

        S <- T_i @ S + S_i      for i = 1 .. n

    Args:
        prefix_state: running state before the segments, shape [..., K, V].
        transitions: stacked per-segment T_C, shape [n, ..., K, K].
        zero_start_states: stacked per-segment S_{C|0}, shape [n, ..., K, V].

    Returns:
        The composed state, same shape as ``prefix_state``.

    Note:
        For models applying RoPE inside the linear layer with a scalar
        transition operator (e.g. Ring-2.5), each zero_start_state must be
        re-rotated to its global start position BEFORE calling this function
        (S_{C|p} = R(p) S_{C|0}); Qwen3.5 (GDN, no in-layer RoPE) needs no
        such adjustment.
    """
    if transitions.shape[0] != zero_start_states.shape[0]:
        raise ValueError(
            f"segment count mismatch: {transitions.shape[0]} transitions vs "
            f"{zero_start_states.shape[0]} states"
        )
    result = prefix_state
    for t_c, s_c in zip(transitions, zero_start_states):
        result = torch.matmul(t_c, result) + s_c
    return result
