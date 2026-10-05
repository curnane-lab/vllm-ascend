# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim — the canonical binding lives in vllm_ascend.

The AscendC LeapQuant flush binding was integrated into
``vllm_ascend.ops.gdn_lq_flush_ascendc`` (shared-workspace executor caching,
IgnoreContiguous-compatible strided layout support). This shim keeps older
imports working.
"""

from vllm_ascend.ops.gdn_lq_flush_ascendc import (  # noqa: F401
    flush_pools_ascendc,
    gdn_lq_flush_ascendc,
)
