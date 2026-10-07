"""M5 base: pools.reconstruct_dense / quantize_from_tensor contracts.

- quantize_from_tensor writes the same checkpoint as init_from_dense for
  the same dense input (it replaces the dense slot write on the prefill
  path once slots hold the composite format only)
- reconstruct_dense matches decode_state on a fresh slot (window empty)
- both bump _lq_epoch (the due-mirror invalidation hook, E5)
"""

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "vllm-ascend"))
sys.path.insert(0, str(_REPO / "leapquant-npu"))

from vllm_ascend.ops.gdn_leapquant import GDNLeapQuantPools  # noqa: E402

DEVICE = "npu"
H, V, K = 32, 128, 128


def _mk(slots, t, seed):
    gen = torch.Generator().manual_seed(seed)
    idx = (torch.arange(t) + 1).to(DEVICE)
    buf = torch.randn(slots, H, V, K, generator=gen).to(DEVICE) * 0.05
    return buf, idx


def main():
    buf, idx = _mk(16, 4, 5)
    dense = buf[idx].clone()

    p1 = GDNLeapQuantPools(buf.clone())
    p1.init_from_dense(p1._state, idx)
    p2 = GDNLeapQuantPools(buf.clone())
    p2.quantize_from_tensor(dense, idx)
    for nm in ("codes", "s_k", "s_v", "u", "q", "w", "p", "fill"):
        a = getattr(p1, nm)[idx].float()
        b = getattr(p2, nm)[idx].float()
        assert torch.equal(a, b), f"{nm} mismatch"
    assert p1._lq_epoch == 1 and p2._lq_epoch == 1, (p1._lq_epoch, p2._lq_epoch)
    print("M5 BASE T1 PASS: quantize_from_tensor == init_from_dense, epoch bumped")

    # reconstruct_dense == decode_state on a fresh slot (window empty, w=0)
    r = p2.reconstruct_dense(idx)
    d = p2.decode_state(idx)
    rel = ((r - d).norm() / d.norm().clamp_min(1e-9)).item()
    assert rel < 1e-5, rel
    # and it approximates the original dense within quantization error
    rel2 = ((r - dense.float()).norm() / dense.float().norm()).item()
    assert rel2 < 0.1, rel2
    print(f"M5 BASE T2 PASS: reconstruct==decode_state ({rel:.2e}); "
          f"vs dense rel err {rel2:.4f}")

    # epoch bumps on every call (mirror invalidation contract)
    e0 = p2._lq_epoch
    p2.quantize_from_tensor(dense, idx)
    assert p2._lq_epoch == e0 + 1
    print("M5 BASE T3 PASS: epoch increments per quantize")
    print("M5 BASE ALL PASS")


if __name__ == "__main__":
    main()
