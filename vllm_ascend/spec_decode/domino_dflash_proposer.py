from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class AscendDominoDflashProposer(AscendDflashProposer):
    """NPU DFlash proposer with the Domino causal correction head.

    This class extends :class:`AscendDflashProposer` with Domino-specific
    configuration parsing and (in a follow-up change) sequential GRU sampling.
    The parallel DFlash backbone and context-KV precomputation are inherited
    unchanged.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)

        draft_hf_config = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        dflash_config = getattr(draft_hf_config, "dflash_config", {}) or {}
        projector_type = dflash_config.get("projector_type", None)
        if projector_type != "domino":
            raise ValueError(
                "AscendDominoDflashProposer requires "
                f"dflash_config.projector_type='domino', got {projector_type!r}"
            )

        self.pure_draft_prefix_len = int(
            dflash_config.get("pure_draft_prefix_len", 0)
        )
        self.gru_hidden_dim = int(dflash_config["gru_hidden_dim"])

    def _get_model(self) -> Any:
        # Load the same DFlashQwen3 model as the base proposer.
        # The Domino head is part of the shared vllm model layer.
        return super()._get_model()
