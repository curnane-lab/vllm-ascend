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
        self.shift_label = bool(dflash_config.get("shift_label", False))
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

        # ``num_input_tokens`` equals batch_size * num_query_per_req only on the
        # real propose path. ACL-graph profiling / dummy_run can pass an
        # arbitrary capture size (e.g. ``num_input_tokens != batch * (1+nspec)``).
        # In that case the backbone forward is sufficient for memory/profile
        # accounting; we skip the sequential Domino sampling and return zeros.
        num_query_per_req = 1 + self.num_speculative_tokens
        expected = batch_size * num_query_per_req
        if num_input_tokens != expected or last_hidden_states.shape[0] < expected:
            return torch.zeros(
                batch_size,
                self.num_speculative_tokens,
                dtype=torch.int64,
                device=self.device,
            )

        query_hidden_states = last_hidden_states[:expected]
        hidden_3d = query_hidden_states.view(
            batch_size, num_query_per_req, self.hidden_size
        )
        input_ids_2d = self.input_ids[:expected].view(
            batch_size, num_query_per_req
        )

        # Pre-compute base_logits for all speculative steps in a single batched
        # GEMM.  This replaces num_spec separate compute_logits calls (each a
        # small [1, H] → [1, vocab] GEMM) with one [B*nspec, H] → [B*nspec,
        # vocab] GEMM, dramatically reducing kernel-launch overhead on NPU.
        pos_offset = 0 if self.shift_label else 1
        nspec = self.num_speculative_tokens
        step_indices = torch.arange(
            nspec, device=self.device
        ) + pos_offset  # [nspec]
        spec_hidden = hidden_3d[:, step_indices, :]  # [B, nspec, H]
        base_logits_all = self.model.compute_logits(
            spec_hidden.reshape(batch_size * nspec, self.hidden_size)
        ).view(batch_size, nspec, -1)  # [B, nspec, vocab]

        draft_token_ids = torch.empty(
            batch_size,
            nspec,
            dtype=torch.int64,
            device=self.device,
        )

        # Seed all requests' GRU states in one batched call.
        bonus_tokens = input_ids_2d[:, 0]  # [B]
        gru_state = self.model.update_domino_gru_state(bonus_tokens, None)

        for step in range(nspec):
            hidden = spec_hidden[:, step, :]  # [B, H]
            if step < self.pure_draft_prefix_len:
                logits = base_logits_all[:, step, :]
            else:
                bias = self.model.compute_domino_bias(hidden, gru_state)
                logits = base_logits_all[:, step, :] + bias

            tokens = logits.argmax(dim=-1)  # [B]
            draft_token_ids[:, step] = tokens
            gru_state = self.model.update_domino_gru_state(tokens, gru_state)

        return draft_token_ids
