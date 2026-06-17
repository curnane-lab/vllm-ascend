from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class AscendDominoDflashProposer(AscendDflashProposer):
    """NPU DFlash proposer with the Domino causal correction head.

    The parallel DFlash backbone forward is reused, but token sampling is
    performed sequentially: the first ``pure_draft_prefix_len`` speculative
    tokens are sampled from the backbone logits, and subsequent tokens use
    GRU-conditioned logit corrections.

    Because Domino sampling is inherently sequential, ACL graphs are disabled
    for this proposer; the backbone runs eagerly together with the GRU head.
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

        # Sequential GRU sampling cannot be captured by the ACL graph.
        self.use_cuda_graph = False

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        """Run one Domino speculative decoding step.

        The DFlash backbone is executed once in parallel for all query positions,
        then tokens are sampled sequentially with the Domino correction head.
        """
        model_kwargs = self.build_model_inputs_first_pass(num_input_tokens)

        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
        else:
            last_hidden_states, _ = ret_hidden_states

        # ``num_input_tokens`` equals batch_size * num_query_per_req for DFlash.
        num_query_per_req = 1 + self.num_speculative_tokens
        query_hidden_states = last_hidden_states[: batch_size * num_query_per_req]
        hidden_3d = query_hidden_states.view(
            batch_size, num_query_per_req, self.hidden_size
        )
        input_ids_2d = self.input_ids[: batch_size * num_query_per_req].view(
            batch_size, num_query_per_req
        )

        draft_token_ids = torch.zeros(
            batch_size,
            self.num_speculative_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        for req_idx in range(batch_size):
            # The bonus token at query position 0 seeds the Domino GRU state.
            bonus_token = input_ids_2d[req_idx, 0]
            gru_state = self.model.update_domino_gru_state(
                bonus_token.unsqueeze(0), None
            )

            for step in range(self.num_speculative_steps):
                query_pos = step + 1
                hidden = hidden_3d[req_idx, query_pos, :].unsqueeze(0)

                if step < self.pure_draft_prefix_len:
                    logits = self.model.compute_logits(hidden)
                else:
                    logits, gru_state = self.model.compute_domino_logits(
                        hidden, gru_state
                    )

                token = logits.argmax(dim=-1).view(())
                draft_token_ids[req_idx, step] = token

                # Update the GRU state with the sampled token for the next step.
                gru_state = self.model.update_domino_gru_state(
                    token.unsqueeze(0), gru_state
                )

        return draft_token_ids
