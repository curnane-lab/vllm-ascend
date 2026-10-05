"""LeapQuant GDN prototypes (torch reference + AscendC bindings)."""

try:  # reference/sim prototypes live outside this NPU workspace copy
    from .reference import gdn_step, gdn_flush, decode_checkpoint, make_checkpoint
    from .sim import GDNConfig, gen_token_inputs, oracle_rollout, leapquant_rollout

    __all__ = [
        "gdn_step",
        "gdn_flush",
        "decode_checkpoint",
        "make_checkpoint",
        "GDNConfig",
        "gen_token_inputs",
        "oracle_rollout",
        "leapquant_rollout",
    ]
except ImportError:  # pragma: no cover - optional in the NPU workspace
    __all__ = []
