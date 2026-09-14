# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Smoke coverage for hybrid (GDN + full attention) models running with
``--mamba-ssm-cache-dtype bfloat16``.

Halving the ssm page moves the per-block K-page == ssm-page interlock from
1024 to 512 tokens, so the engine default block under bf16 lands on the
byte-identical legacy pool layout (see
tests/ut/patch/platform/test_patch_mamba_config_guard.py, "bf16 ssm state"
section, for the pinned geometry map). This e2e checks the engine accepts
the dtype override, derives the expected block geometry, and generates
coherently on the NPU mamba kernels with bf16 ssm pages.
"""

from tests.e2e.conftest import VllmRunner


def test_qwen3_next_ssm_bf16_mp_tp4():
    example_prompts = [
        "Hello, my name is",
    ] * 4
    max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        mamba_ssm_cache_dtype="bfloat16",
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]},
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
        del vllm_model
