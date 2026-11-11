from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from sparsegdn_reference import GDNInputs
from sparsegdn_vectorized import run_vectorized_reference


class SparseGDNWorker:
    pass


def _install_sparse_gdn_scope_a() -> None:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops import gdn as ascend_gdn

    mode = os.environ.get("SPARSEGDN_MODE", "sparse")
    tag = os.environ.get("SPARSEGDN_RUN_TAG", "unnamed")
    trace_dir = Path(os.environ.get("SPARSEGDN_TRACE_DIR", "/tmp/sparsegdn_trace"))
    prefix_tokens = int(os.environ["SPARSEGDN_PREFIX_TOKENS"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    active = {"prefix": None, "z": None, "norm": None}
    request_state = {"index": -1, "tag": tag}
    original_forward = ascend_gdn.AscendGatedDeltaNetAttention.forward
    original_logits = Qwen3_5ForCausalLMBase.compute_logits

    def projection_hook(layer, projected):
        if isinstance(projected, tuple):
            projected = projected[0]
        if hasattr(layer, "in_proj_qkv"):
            z = projected
        else:
            qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
            z = projected[..., qkv_size:]
        active["z"] = z.reshape(z.size(0), -1, layer.head_v_dim)
        active["norm"] = layer.norm.weight
        active["prefix"] = layer.prefix

    def forward_with_projection_capture(self, hidden_states, output=None):
        if not getattr(self, "_sparse_gdn_projection_hook", False):
            projection = self.in_proj_z if hasattr(self, "in_proj_qkv") else self.in_proj_qkvz

            def hook(module, args, result):
                projection_hook(self, result)

            projection.register_forward_hook(hook)
            self._sparse_gdn_projection_hook = True
        return original_forward(self, hidden_states, output)

    def capture_logits(self, hidden_states, *args, **kwargs):
        logits = original_logits(self, hidden_states, *args, **kwargs)
        if logits is not None:
            torch.save(
                logits.detach().float().cpu(),
                trace_dir / f"logits_{request_state['tag']}_{active['prefix'] or 'unknown'}.pt",
            )
        return logits

    ascend_gdn.AscendGatedDeltaNetAttention.forward = forward_with_projection_capture
    Qwen3_5ForCausalLMBase.compute_logits = capture_logits
    if mode == "dense":
        return
    if mode != "sparse":
        raise ValueError(f"unknown SPARSEGDN_MODE: {mode}")

    def unsupported_dense_fallback(*args, **kwargs):
        raise RuntimeError("SparseGDN SCOPE-A reference did not reach its fused wrapper")

    ascend_gdn.chunk_gated_delta_rule = unsupported_dense_fallback

    def sparse_scope_a(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        scale,
    ):
        if active["z"] is None or active["norm"] is None:
            raise RuntimeError("current-layer Z/RMSNorm capture is missing")
        boundaries = cu_seqlens.detach().cpu().tolist()
        if len(boundaries) != 2 or boundaries[0] != 0 or boundaries[1] != q.shape[1]:
            raise RuntimeError("SCOPE-A accepts exactly one complete prompt")
        if q.shape[1] != active["z"].shape[0]:
            raise RuntimeError("SCOPE-A does not accept mixed decode/prefill or chunks")
        if initial_state.float().abs().sum().item() != 0.0:
            raise RuntimeError("SCOPE-A requires zero initial state")
        if abs(scale - q.shape[-1] ** -0.5) > 1e-12:
            raise RuntimeError("query scale does not match the reference contract")
        query = l2norm_fwd(q.contiguous()).squeeze(0).float().cpu()
        key = l2norm_fwd(k.contiguous()).squeeze(0).float().cpu()
        value = v.squeeze(0).float().cpu()
        inputs = GDNInputs(
            query,
            key,
            value,
            g.squeeze(0).float().cpu(),
            beta.squeeze(0).float().cpu(),
            active["z"].float().cpu(),
            active["norm"].float().cpu(),
            None,
        )
        result = run_vectorized_reference(inputs, return_state_trace=True)
        boundary_chunk = prefix_tokens // 64
        if boundary_chunk >= result.state_trace.shape[0]:
            raise RuntimeError("prefix boundary is outside the current prompt")
        if ".layers.0." in (active["prefix"] or ""):
            request_state["index"] += 1
            request_state["tag"] = f"{tag}_request{request_state['index']}"
        layer_tag = f"{request_state['tag']}_{active['prefix'] or 'unknown'}"
        if not (result.output.isfinite().all() and result.final_state.isfinite().all()):
            torch.save(
                {
                    "inputs": inputs,
                    "mask": result.selector.mask.cpu(),
                    "scores": result.intra.scores.cpu(),
                    "U": result.intra.u.cpu(),
                    "corrected_U": result.corrected_u.cpu(),
                    "output": result.output.cpu(),
                    "state_at_prefix": result.state_trace[boundary_chunk].cpu(),
                    "final_state": result.final_state.cpu(),
                },
                trace_dir / f"invalid_{layer_tag}.pt",
            )
            raise FloatingPointError(f"non-finite SparseGDN result in {active['prefix']}")
        mask = result.selector.mask.cpu()
        tiers = result.selector.tiers.cpu()
        torch.save(
            {
                "layer": active["prefix"],
                "prompt_tokens": inputs.length,
                "prefix_tokens": prefix_tokens,
                "mask": mask,
                "tiers": tiers,
                "scores": result.intra.scores.cpu(),
                "U": result.intra.u.bfloat16().cpu(),
                "output": result.output.bfloat16().cpu(),
                "state_at_prefix": result.state_trace[boundary_chunk].cpu(),
                "final_state": result.final_state.cpu(),
            },
            trace_dir / f"layer_{layer_tag}.pt",
        )
        summary = {
            "layer": active["prefix"],
            "prompt_tokens": inputs.length,
            "prefix_tokens": prefix_tokens,
            "valid_events": result.selector.valid_events,
            "capacity": result.selector.capacity,
            "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
            "tiers": tiers.tolist(),
            "U_frobenius": result.intra.u.square().sum().sqrt().item(),
            "output_frobenius": result.output.square().sum().sqrt().item(),
            "state_at_prefix_frobenius": result.state_trace[boundary_chunk].square().sum().sqrt().item(),
            "final_state_frobenius": result.final_state.square().sum().sqrt().item(),
            "finite": bool(result.output.isfinite().all() and result.final_state.isfinite().all()),
        }
        with (trace_dir / f"summary_{request_state['tag']}.jsonl").open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
        output = result.output.to(q.device, dtype=q.dtype).unsqueeze(0)
        final_state = result.final_state.transpose(-1, -2).to(q.device, dtype=initial_state.dtype)
        return output, final_state.unsqueeze(0)

    ascend_gdn.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused = staticmethod(sparse_scope_a)
    ascend_gdn.AscendGatedDeltaNetAttention._probe_fused_chunk = classmethod(lambda cls: True)

    def fla_sparse_scope_a(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale,
        prebuilt_meta,
        fused_fwd,
    ):
        cu_seqlens = torch.tensor(prebuilt_meta.cu_seqlens_host, dtype=torch.int64)
        return sparse_scope_a(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            scale,
        )

    DeviceOperator.fla_gdn_prefill = staticmethod(fla_sparse_scope_a)


_install_sparse_gdn_scope_a()
