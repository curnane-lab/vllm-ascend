from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

CHUNK_SIZE = 64
BLOCK_SIZE = 16
TAIL_SIZE = 16
RETENTION_RATES = (0.08, 0.30, 0.95)
TIER_NAMES = ("L", "M", "H")
BLOCK_ORDER = ((0, 0), (1, 1), (2, 2), (3, 3), (1, 0), (2, 1), (3, 2))


@dataclass
class GDNInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor
    z: torch.Tensor
    norm_scale: torch.Tensor
    initial_state: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return self.q.shape[0]

    @property
    def num_value_heads(self) -> int:
        return self.v.shape[1]


@dataclass
class PaddedInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor
    z: torch.Tensor
    g: torch.Tensor
    gamma: torch.Tensor
    initial_state: torch.Tensor
    valid_lengths: torch.Tensor
    num_chunks: int
    original_length: int


@dataclass
class IntraOutputs:
    u: torch.Tensor
    read_coefficients: torch.Tensor
    scores: torch.Tensor
    solve: str


@dataclass
class SelectorOutputs:
    mask: torch.Tensor
    tiers: torch.Tensor
    normalized_scores: torch.Tensor
    gate_energy: torch.Tensor
    lifetime: torch.Tensor
    visibility: torch.Tensor
    upgrade_gains: torch.Tensor
    normalized_upgrade_gains: torch.Tensor
    pair_order: torch.Tensor
    event_offsets: torch.Tensor
    event_indices: torch.Tensor
    group_event_map: torch.Tensor
    decay_prefix: torch.Tensor
    keep_counts: tuple[int, int, int]
    split_counts: tuple[int, int]
    valid_events: int
    capacity: int


@dataclass
class ReferenceOutputs:
    output: torch.Tensor
    final_state: torch.Tensor
    state_trace: torch.Tensor | None
    corrected_u: torch.Tensor
    intra: IntraOutputs
    selector: SelectorOutputs


@dataclass
class StreamingOutputs:
    output: torch.Tensor
    final_state: torch.Tensor
    corrected_u: torch.Tensor
    boundary_states: dict[int, torch.Tensor]
    boundary_outputs: dict[int, torch.Tensor]


def make_inputs(
    length: int,
    seed: int,
    key_dim: int = 8,
    value_dim: int | None = None,
    num_key_heads: int = 16,
    num_value_heads: int = 32,
    dtype: torch.dtype = torch.float64,
    with_initial_state: bool = False,
) -> GDNInputs:
    value_dim = value_dim or key_dim
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(length, num_key_heads, key_dim, generator=generator, dtype=dtype)
    k = torch.randn(length, num_key_heads, key_dim, generator=generator, dtype=dtype)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    v = torch.randn(
        length,
        num_value_heads,
        value_dim,
        generator=generator,
        dtype=dtype,
    )
    log_decay = -0.2 * torch.rand(
        length,
        num_value_heads,
        generator=generator,
        dtype=dtype,
    )
    beta = torch.rand(length, num_value_heads, generator=generator, dtype=dtype)
    z = torch.randn(
        length,
        num_value_heads,
        value_dim,
        generator=generator,
        dtype=dtype,
    )
    norm_scale = torch.randn(value_dim, generator=generator, dtype=dtype)
    initial_state = None
    if with_initial_state:
        initial_state = torch.randn(
            num_value_heads,
            key_dim,
            value_dim,
            generator=generator,
            dtype=dtype,
        )
    return GDNInputs(q, k, v, log_decay, beta, z, norm_scale, initial_state)


def slice_inputs(inputs: GDNInputs, start: int, end: int) -> GDNInputs:
    return GDNInputs(
        inputs.q[start:end],
        inputs.k[start:end],
        inputs.v[start:end],
        inputs.log_decay[start:end],
        inputs.beta[start:end],
        inputs.z[start:end],
        inputs.norm_scale,
        None,
    )


def concatenate_inputs(parts: Sequence[GDNInputs]) -> GDNInputs:
    if not parts:
        raise ValueError("at least one input part is required")
    initial_state = parts[0].initial_state
    return GDNInputs(
        *(
            torch.cat([getattr(part, name) for part in parts], dim=0)
            for name in ("q", "k", "v", "log_decay", "beta", "z")
        ),
        parts[0].norm_scale,
        initial_state,
    )


def pad_inputs(inputs: GDNInputs) -> PaddedInputs:
    length = inputs.length
    if length == 0:
        raise ValueError("empty sequences are outside this reference contract")
    num_chunks = math.ceil(length / CHUNK_SIZE)
    padded_length = num_chunks * CHUNK_SIZE
    padding = padded_length - length
    q = torch.nn.functional.pad(inputs.q, (0, 0, 0, 0, 0, padding))
    k = torch.nn.functional.pad(inputs.k, (0, 0, 0, 0, 0, padding))
    v = torch.nn.functional.pad(inputs.v, (0, 0, 0, 0, 0, padding))
    log_decay = torch.nn.functional.pad(inputs.log_decay, (0, 0, 0, padding))
    beta = torch.nn.functional.pad(inputs.beta, (0, 0, 0, padding))
    z = torch.nn.functional.pad(inputs.z, (0, 0, 0, 0, 0, padding))
    g = log_decay.view(num_chunks, CHUNK_SIZE, -1).cumsum(dim=1)
    gamma = g[:, -1]
    valid_lengths = torch.full((num_chunks,), CHUNK_SIZE, dtype=torch.long, device=q.device)
    valid_lengths[-1] = length - (num_chunks - 1) * CHUNK_SIZE
    if inputs.initial_state is None:
        initial_state = torch.zeros(
            inputs.v.shape[1],
            inputs.k.shape[-1],
            inputs.v.shape[-1],
            dtype=v.dtype,
            device=v.device,
        )
    else:
        initial_state = inputs.initial_state
    return PaddedInputs(
        q,
        k,
        v,
        log_decay,
        beta,
        z,
        g,
        gamma,
        initial_state,
        valid_lengths,
        num_chunks,
        length,
    )


def _value_head_keys(k: torch.Tensor, num_value_heads: int) -> torch.Tensor:
    if k.shape[-2] * 2 != num_value_heads:
        raise ValueError("this contract requires exactly two value heads per key head")
    return k.repeat_interleave(2, dim=-2)


def _delta_matrix(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    interactions = k @ k.transpose(0, 1)
    strictly_lower = torch.tril(torch.ones_like(interactions, dtype=torch.bool), diagonal=-1)
    exponent = torch.where(strictly_lower, g[:, None] - g[None, :], torch.zeros_like(interactions))
    decay = torch.exp2(exponent)
    return torch.eye(k.shape[0], dtype=k.dtype, device=k.device) + beta[:, None] * interactions * decay * strictly_lower


def _block_coefficients(
    matrix: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    diagonal = []
    adjacent = []
    for block in range(4):
        current = slice(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE)
        diagonal.append(torch.linalg.inv(matrix[current, current]))
    for block in range(1, 4):
        current = slice(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE)
        previous = slice((block - 1) * BLOCK_SIZE, block * BLOCK_SIZE)
        adjacent.append(-diagonal[block] @ matrix[current, previous] @ diagonal[block - 1])
    return diagonal, adjacent


def _intra_chunk(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    solve: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _delta_matrix(k, g, beta)
    source_scale = beta * torch.exp2(g)
    if solve == "full64":
        coefficients = torch.linalg.inv(matrix)
        u = coefficients @ (beta[:, None] * v)
        read_coefficients = coefficients * source_scale[None, :]
        return u, read_coefficients
    if solve != "block-adjacent":
        raise ValueError(f"unknown solve: {solve}")
    diagonal, adjacent = _block_coefficients(matrix)
    blocks = diagonal + adjacent
    read_blocks = []
    u_blocks = []
    for target, source in BLOCK_ORDER:
        source_slice = slice(source * BLOCK_SIZE, (source + 1) * BLOCK_SIZE)
        coefficient = blocks[len(read_blocks)]
        read_blocks.append(coefficient * source_scale[source_slice][None, :])
        value_contribution = coefficient @ (beta[source_slice, None] * v[source_slice])
        if target == source:
            u_blocks.insert(target, value_contribution)
        else:
            u_blocks[target] = u_blocks[target] + value_contribution
    return torch.cat(u_blocks, dim=0), torch.stack(read_blocks)


def compute_intra(
    padded: PaddedInputs,
    solve: str,
) -> tuple[PaddedInputs, IntraOutputs]:
    chunk_count = padded.num_chunks
    head_count = padded.v.shape[-2]
    value_dim = padded.v.shape[-1]
    k = _value_head_keys(padded.k, head_count)
    u = torch.empty(
        chunk_count,
        CHUNK_SIZE,
        head_count,
        value_dim,
        dtype=padded.v.dtype,
        device=padded.v.device,
    )
    coefficient_shape = (64, 64) if solve == "full64" else (7, 16, 16)
    read_coefficients = torch.empty(
        chunk_count,
        head_count,
        *coefficient_shape,
        dtype=padded.v.dtype,
        device=padded.v.device,
    )
    for chunk in range(chunk_count):
        for head in range(head_count):
            u_chunk, coefficients = _intra_chunk(
                k[chunk, :, head],
                padded.v[chunk, :, head],
                padded.g[chunk, :, head],
                padded.beta[chunk, :, head],
                solve,
            )
            u[chunk, :, head] = u_chunk
            read_coefficients[chunk, head] = coefficients
    scores = u[:, -TAIL_SIZE:].square().sum(dim=(1, 3)).sqrt()
    return padded, IntraOutputs(u, read_coefficients, scores, solve)


def _keep_counts(num_chunks: int) -> tuple[int, int, int]:
    if num_chunks == 1:
        return (1, 1, 1)
    return tuple(max(2, min(num_chunks, round(rate * num_chunks))) for rate in RETENTION_RATES)


def _candidate_sets(
    normalized_scores: torch.Tensor,
    pair: int,
    num_chunks: int,
) -> list[set[int]]:
    if num_chunks == 1:
        return [{0}, {0}, {0}]
    ranked = sorted(
        range(1, num_chunks - 1),
        key=lambda chunk: (-float(normalized_scores[chunk, pair]), chunk),
    )
    result = []
    for count in _keep_counts(num_chunks):
        result.append({0, num_chunks - 1, *ranked[: count - 2]})
    return result


def _gate_energy(inputs: GDNInputs, padded: PaddedInputs) -> torch.Tensor:
    energy = torch.zeros(inputs.num_value_heads, dtype=padded.v.dtype, device=padded.v.device)
    value_dim = inputs.v.shape[-1]
    for chunk in range(padded.num_chunks):
        for block in range(4):
            token = chunk * CHUNK_SIZE + 8 + block * BLOCK_SIZE
            if token >= inputs.length:
                continue
            coordinate = 73 * (4 * chunk + block) % value_dim
            sampled = inputs.z[token, :, coordinate]
            gated = inputs.norm_scale[coordinate] * torch.nn.functional.silu(sampled)
            energy += gated.square()
    return energy / (4 * padded.num_chunks)


def _split_score(ordered_gains: Sequence[float], count: int) -> float:
    total = len(ordered_gains)
    upper = sum(ordered_gains[:count]) / count
    lower = sum(ordered_gains[count:]) / (total - count)
    return count * (total - count) / total * (upper - lower) ** 2


def _choose_split(
    pair_indices: Iterable[int],
    gains: Sequence[float],
    candidates: Sequence[int],
    fallback: int,
) -> tuple[int, list[int]]:
    ordered = sorted(pair_indices, key=lambda pair: (-gains[pair], pair))
    if not ordered:
        return 0, ordered
    values = [float(gains[pair]) for pair in ordered]
    if values[0] == values[-1]:
        return fallback, ordered
    best_count = None
    best_score = None
    for count in candidates:
        if count <= 0 or count >= len(ordered):
            continue
        score = _split_score(values, count)
        if best_score is None or score > best_score:
            best_count, best_score = count, score
        elif score == best_score:
            current_key = (abs(count - fallback), count)
            best_key = (abs(best_count - fallback), best_count)
            if current_key < best_key:
                best_count = count
    if best_count is None:
        best_count = fallback
    return best_count, ordered


def _selector_from_mask(
    mask: torch.Tensor,
    tiers: torch.Tensor,
    gamma: torch.Tensor,
    scores: torch.Tensor,
    normalized_scores: torch.Tensor,
    gate_energy: torch.Tensor,
    lifetime: torch.Tensor,
    visibility: torch.Tensor,
    upgrade_gains: torch.Tensor,
    normalized_upgrade_gains: torch.Tensor,
) -> SelectorOutputs:
    pair_count, num_chunks = mask.shape
    pair_order = torch.tensor(
        [pair for tier in (2, 1, 0) for pair in range(pair_count) if int(tiers[pair]) == tier],
        dtype=torch.long,
        device=mask.device,
    )
    event_indices = []
    event_offsets = [0]
    group_event_map = torch.full(
        (pair_count, math.ceil(num_chunks / 2)),
        -1,
        dtype=torch.long,
        device=mask.device,
    )
    for pair in pair_order.tolist():
        events = torch.nonzero(mask[pair], as_tuple=False).flatten().tolist()
        event_indices.extend(events)
        event_offsets.append(len(event_indices))
    for pair in range(pair_count):
        row = pair_order.tolist().index(pair)
        latest = -1
        event_position = 0
        events = torch.nonzero(mask[pair], as_tuple=False).flatten().tolist()
        for group in range(math.ceil(num_chunks / 2)):
            group_start = 2 * group
            while event_position < len(events) and events[event_position] < group_start:
                latest = event_offsets[row] + event_position
                event_position += 1
            group_event_map[pair, group] = latest
    decay_prefix = torch.cat([torch.zeros_like(gamma[:1]), gamma.cumsum(dim=0)], dim=0)
    counts = _keep_counts(num_chunks)
    split_counts = tuple(int((tiers == tier).sum()) for tier in (2, 1))
    return SelectorOutputs(
        mask,
        tiers,
        normalized_scores,
        gate_energy,
        lifetime,
        visibility,
        upgrade_gains,
        normalized_upgrade_gains,
        pair_order,
        torch.tensor(event_offsets, dtype=torch.long, device=mask.device),
        torch.tensor(event_indices, dtype=torch.long, device=mask.device),
        group_event_map,
        decay_prefix,
        counts,
        split_counts,
        int(mask.sum()),
        pair_count * counts[2],
    )


def select_events(
    inputs: GDNInputs,
    padded: PaddedInputs,
    intra: IntraOutputs,
) -> SelectorOutputs:
    num_chunks = padded.num_chunks
    head_count = inputs.num_value_heads
    pair_count = head_count // 2
    scores = intra.scores
    norms = scores.norm(dim=0, keepdim=True)
    per_head = torch.where(norms > 0, scores / norms, torch.zeros_like(scores))
    normalized_scores = per_head.view(num_chunks, pair_count, 2).mean(dim=-1)
    gate_energy = _gate_energy(inputs, padded)
    mean_energy = gate_energy.mean()
    visibility = torch.where(
        mean_energy > 0,
        (gate_energy / mean_energy).sqrt(),
        torch.zeros_like(gate_energy),
    )
    retention = torch.exp2(padded.gamma)
    mean_retention = retention[-min(16, num_chunks) :].mean(dim=0)
    lifetime = torch.stack(
        [
            torch.tensor(
                sum(float(mean_retention[h]) ** j for j in range(1, num_chunks - 1)),
                dtype=scores.dtype,
                device=scores.device,
            )
            for h in range(head_count)
        ]
    )
    candidates = [
        [_candidate_sets(normalized_scores, pair, num_chunks) for _ in range(3)] for pair in range(pair_count)
    ]
    candidates = [entry[0] for entry in candidates]
    upgrade_gains = torch.zeros(pair_count, 2, dtype=scores.dtype, device=scores.device)
    for pair in range(pair_count):
        for upgrade, (lower, upper) in enumerate(((0, 1), (1, 2))):
            additional = candidates[pair][upper] - candidates[pair][lower]
            for chunk in additional:
                for head in (2 * pair, 2 * pair + 1):
                    upgrade_gains[pair, upgrade] += scores[chunk, head] * lifetime[head] * visibility[head]
    normalized_gains = torch.zeros_like(upgrade_gains)
    for upgrade in range(2):
        mean_gain = upgrade_gains[:, upgrade].mean()
        if float(mean_gain) != 0.0:
            normalized_gains[:, upgrade] = upgrade_gains[:, upgrade] / mean_gain
    if num_chunks == 1:
        mask = torch.ones(pair_count, 1, dtype=torch.bool, device=scores.device)
        tiers = torch.zeros(pair_count, dtype=torch.long, device=scores.device)
        return _selector_from_mask(
            mask,
            tiers,
            padded.gamma,
            scores,
            normalized_scores,
            gate_energy,
            lifetime,
            visibility,
            upgrade_gains,
            normalized_gains,
        )
    high_candidates = (1, 2, 3, 4)
    high_count, high_order = _choose_split(range(pair_count), normalized_gains[:, 1], high_candidates, 3)
    high_pairs = set(high_order[:high_count])
    remaining = [pair for pair in range(pair_count) if pair not in high_pairs]
    medium_candidates = {
        1: (11, 12, 13),
        2: (7, 8, 9),
        3: (3, 4, 5),
        4: (0,),
    }[high_count]
    medium_fallback = 4 * (4 - high_count)
    medium_count, medium_order = _choose_split(
        remaining,
        normalized_gains[:, 0],
        medium_candidates,
        medium_fallback,
    )
    medium_pairs = set(medium_order[:medium_count])
    tiers = torch.zeros(pair_count, dtype=torch.long, device=scores.device)
    for pair in medium_pairs:
        tiers[pair] = 1
    for pair in high_pairs:
        tiers[pair] = 2
    mask = torch.zeros(pair_count, num_chunks, dtype=torch.bool, device=scores.device)
    for pair in range(pair_count):
        selected = candidates[pair][int(tiers[pair])]
        mask[pair, sorted(selected)] = True
    return _selector_from_mask(
        mask,
        tiers,
        padded.gamma,
        scores,
        normalized_scores,
        gate_energy,
        lifetime,
        visibility,
        upgrade_gains,
        normalized_gains,
    )


def full_selection(
    padded: PaddedInputs,
    intra: IntraOutputs,
) -> SelectorOutputs:
    pair_count = padded.v.shape[-2] // 2
    mask = torch.ones(
        pair_count,
        padded.num_chunks,
        dtype=torch.bool,
        device=padded.v.device,
    )
    tiers = torch.full((pair_count,), 2, dtype=torch.long, device=padded.v.device)
    zeros = torch.zeros_like(intra.scores)
    gain_zeros = torch.zeros(pair_count, 2, dtype=zeros.dtype, device=zeros.device)
    return _selector_from_mask(
        mask,
        tiers,
        padded.gamma,
        intra.scores,
        zeros,
        zeros[0],
        zeros[0],
        zeros[0],
        gain_zeros,
        gain_zeros,
    )


def frozen_selection(
    mask: torch.Tensor,
    tiers: torch.Tensor,
    padded: PaddedInputs,
    intra: IntraOutputs,
) -> SelectorOutputs:
    zeros = torch.zeros_like(intra.scores)
    pair_count = mask.shape[0]
    gain_zeros = torch.zeros(pair_count, 2, dtype=zeros.dtype, device=zeros.device)
    return _selector_from_mask(
        mask,
        tiers,
        padded.gamma,
        intra.scores,
        zeros,
        zeros[0],
        zeros[0],
        zeros[0],
        gain_zeros,
        gain_zeros,
    )


def _materialize_w(
    padded: PaddedInputs,
    intra: IntraOutputs,
    chunk: int,
    head: int,
) -> torch.Tensor:
    k = _value_head_keys(padded.k, padded.v.shape[-2])[chunk, :, head]
    coefficients = intra.read_coefficients[chunk, head]
    if intra.solve == "full64":
        return coefficients @ k
    w_blocks = []
    for target in range(4):
        target_slice = slice(target * BLOCK_SIZE, (target + 1) * BLOCK_SIZE)
        value = coefficients[target] @ k[target_slice]
        if target:
            previous_slice = slice((target - 1) * BLOCK_SIZE, target * BLOCK_SIZE)
            value = value + coefficients[target + 3] @ k[previous_slice]
        w_blocks.append(value)
    return torch.cat(w_blocks, dim=0)


def _local_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    output_g: torch.Tensor,
    source_g: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    exponent = output_g[:, None] - source_g[None, :]
    if causal:
        mask = torch.tril(torch.ones_like(exponent, dtype=torch.bool))
        exponent = torch.where(mask, exponent, torch.zeros_like(exponent))
        return (q @ k.transpose(0, 1)) * torch.exp2(exponent) * mask
    return (q @ k.transpose(0, 1)) * torch.exp2(exponent)


def _output_even(
    padded: PaddedInputs,
    corrected_u: torch.Tensor,
    entry_state: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    k = _value_head_keys(padded.k, padded.v.shape[-2])
    q = _value_head_keys(padded.q, padded.v.shape[-2])
    local = torch.empty_like(corrected_u)
    historical = torch.empty_like(corrected_u)
    for head in range(padded.v.shape[-2]):
        matrix = _local_matrix(
            q[chunk, :, head],
            k[chunk, :, head],
            padded.g[chunk, :, head],
            padded.g[chunk, :, head],
            True,
        )
        local[:, head] = matrix @ corrected_u[:, head]
        historical[:, head] = torch.exp2(padded.g[chunk, :, head, None]) * (q[chunk, :, head] @ entry_state[head])
    return (historical + local) / math.sqrt(padded.k.shape[-1])


def _output_odd(
    padded: PaddedInputs,
    even_u: torch.Tensor,
    odd_u: torch.Tensor,
    entry_state: torch.Tensor,
    even: int,
) -> torch.Tensor:
    odd = even + 1
    k = _value_head_keys(padded.k, padded.v.shape[-2])
    q = _value_head_keys(padded.q, padded.v.shape[-2])
    output = torch.empty_like(odd_u)
    for head in range(padded.v.shape[-2]):
        own = _local_matrix(
            q[odd, :, head],
            k[odd, :, head],
            padded.g[odd, :, head],
            padded.g[odd, :, head],
            True,
        )
        cross_decay = padded.g[odd, :, head, None] + padded.gamma[even, head]
        cross = (q[odd, :, head] @ k[even, :, head].transpose(0, 1)) * torch.exp2(
            cross_decay - padded.g[even, :, head][None, :]
        )
        historical = torch.exp2(cross_decay) * (q[odd, :, head] @ entry_state[head])
        output[:, head] = (historical + cross @ even_u[:, head] + own @ odd_u[:, head]) / math.sqrt(padded.k.shape[-1])
    return output


def _event_step(
    padded: PaddedInputs,
    intra: IntraOutputs,
    mask: torch.Tensor,
    state: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_count = padded.v.shape[-2]
    k = _value_head_keys(padded.k, head_count)
    u = intra.u[chunk].clone()
    head_mask = mask[:, chunk].repeat_interleave(2)
    w = torch.zeros(
        CHUNK_SIZE,
        head_count,
        padded.k.shape[-1],
        dtype=u.dtype,
        device=u.device,
    )
    for head in range(head_count):
        if bool(head_mask[head]):
            w[:, head] = _materialize_w(padded, intra, chunk, head)
    correction = torch.einsum("lhk,hkv->lhv", w, state)
    u = u - correction
    write_weights = torch.exp2(padded.gamma[chunk][None, :] - padded.g[chunk])[:, :, None]
    writes = torch.einsum("lhk,lhv->hkv", k[chunk], u * write_weights) * head_mask[:, None, None]
    decay = torch.exp2(padded.gamma[chunk])[:, None, None]
    state = decay * state + writes
    return u, state


def run_reference(
    inputs: GDNInputs,
    solve: str = "block-adjacent",
    selection: str = "adaptive",
    frozen_mask: torch.Tensor | None = None,
    frozen_tiers: torch.Tensor | None = None,
    return_state_trace: bool = False,
) -> ReferenceOutputs:
    padded = PaddedInputs(**vars(pad_inputs(inputs)))
    padded.q = padded.q.view(padded.num_chunks, CHUNK_SIZE, *padded.q.shape[1:])
    padded.k = padded.k.view(padded.num_chunks, CHUNK_SIZE, *padded.k.shape[1:])
    padded.v = padded.v.view(padded.num_chunks, CHUNK_SIZE, *padded.v.shape[1:])
    padded.log_decay = padded.log_decay.view(padded.num_chunks, CHUNK_SIZE, -1)
    padded.beta = padded.beta.view(padded.num_chunks, CHUNK_SIZE, -1)
    padded.z = padded.z.view(padded.num_chunks, CHUNK_SIZE, *padded.z.shape[1:])
    padded, intra = compute_intra(padded, solve)
    if frozen_mask is not None:
        if frozen_tiers is None:
            raise ValueError("frozen_tiers is required with frozen_mask")
        selector = frozen_selection(frozen_mask, frozen_tiers, padded, intra)
    elif selection == "adaptive":
        selector = select_events(inputs, padded, intra)
    elif selection == "full":
        selector = full_selection(padded, intra)
    else:
        raise ValueError(f"unknown selection: {selection}")
    corrected_u = torch.empty_like(intra.u)
    state = padded.initial_state.clone()
    state_trace = [state.clone()]
    for chunk in range(padded.num_chunks):
        corrected_u[chunk], state = _event_step(padded, intra, selector.mask, state, chunk)
        if return_state_trace:
            state_trace.append(state.clone())
    output = torch.full(
        (padded.num_chunks, CHUNK_SIZE, *inputs.v.shape[1:]),
        float("nan"),
        dtype=inputs.v.dtype,
        device=inputs.v.device,
    )
    for even in range(0, padded.num_chunks, 2):
        entry_state = state_trace[even] if return_state_trace else None
        if entry_state is None:
            entry_state = _reconstruct_entry_state(padded, intra, selector, corrected_u, even)
        output[even] = _output_even(padded, corrected_u[even], entry_state, even)
        if even + 1 < padded.num_chunks:
            output[even + 1] = _output_odd(
                padded,
                corrected_u[even],
                corrected_u[even + 1],
                entry_state,
                even,
            )
    trace = torch.stack(state_trace) if return_state_trace else None
    return ReferenceOutputs(
        output.view(-1, *inputs.v.shape[1:])[: inputs.length],
        state,
        trace,
        corrected_u,
        intra,
        selector,
    )


def _reconstruct_entry_state(
    padded: PaddedInputs,
    intra: IntraOutputs,
    selector: SelectorOutputs,
    corrected_u: torch.Tensor,
    group_start: int,
) -> torch.Tensor:
    state = padded.initial_state.clone()
    for chunk in range(group_start):
        _, state = _event_step(padded, intra, selector.mask, state, chunk)
    return state


def recurrent_dense(inputs: GDNInputs) -> tuple[torch.Tensor, torch.Tensor]:
    state = inputs.initial_state
    if state is None:
        state = torch.zeros(
            inputs.num_value_heads,
            inputs.k.shape[-1],
            inputs.v.shape[-1],
            dtype=inputs.v.dtype,
            device=inputs.v.device,
        )
    else:
        state = state.clone()
    q = _value_head_keys(inputs.q, inputs.num_value_heads)
    k = _value_head_keys(inputs.k, inputs.num_value_heads)
    output = torch.empty_like(inputs.v)
    for token in range(inputs.length):
        decay = torch.exp2(inputs.log_decay[token])[:, None, None]
        read = torch.einsum("hk,hkv->hv", k[token], state)
        update = (
            inputs.beta[token].view(-1, 1, 1)
            * k[token].unsqueeze(-1)
            * (inputs.v[token] - decay[:, :, 0] * read).unsqueeze(1)
        )
        state = decay * state + update
        output[token] = torch.einsum("hk,hkv->hv", q[token], state) / math.sqrt(inputs.k.shape[-1])
    return output, state


def run_frozen_partition(
    inputs: GDNInputs,
    piece_lengths: Sequence[int],
    mask: torch.Tensor,
    tiers: torch.Tensor,
    solve: str = "block-adjacent",
) -> StreamingOutputs:
    if sum(piece_lengths) != inputs.length:
        raise ValueError("piece lengths must sum to the input length")
    if any(length % CHUNK_SIZE for length in piece_lengths[:-1]):
        raise ValueError("this diagnostic driver requires chunk-aligned pieces")
    full = run_reference(
        inputs,
        solve=solve,
        frozen_mask=mask,
        frozen_tiers=tiers,
        return_state_trace=False,
    )
    padded = PaddedInputs(**vars(pad_inputs(inputs)))
    for name in ("q", "k", "v", "log_decay", "beta", "z"):
        value = getattr(padded, name)
        setattr(
            padded,
            name,
            value.view(padded.num_chunks, CHUNK_SIZE, *value.shape[1:]),
        )
    state = padded.initial_state.clone()
    corrected_u = torch.empty_like(full.corrected_u)
    output = torch.full_like(full.corrected_u, float("nan"))
    boundary_states = {}
    boundary_outputs = {}
    pending = None
    token_offset = 0
    chunk_offset = 0
    for piece_index, piece_length in enumerate(piece_lengths):
        piece_chunks = piece_length // CHUNK_SIZE
        if piece_index == len(piece_lengths) - 1 and piece_length % CHUNK_SIZE:
            piece_chunks = math.ceil(piece_length / CHUNK_SIZE)
        for local_chunk in range(piece_chunks):
            chunk = chunk_offset + local_chunk
            entry_state = state.clone()
            corrected_u[chunk], state = _event_step(padded, full.intra, mask, state, chunk)
            if chunk % 2 == 0:
                if pending is not None:
                    raise RuntimeError("an even group was not completed")
                pending = (chunk, entry_state)
            else:
                if pending is None or pending[0] != chunk - 1:
                    raise RuntimeError("missing group-local even-chunk context")
                even, group_state = pending
                output[even] = _output_even(padded, corrected_u[even], group_state, even)
                output[chunk] = _output_odd(
                    padded,
                    corrected_u[even],
                    corrected_u[chunk],
                    group_state,
                    even,
                )
                pending = None
        token_offset += piece_length
        chunk_offset += piece_chunks
        boundary_states[token_offset] = state.clone()
        boundary_outputs[token_offset] = output.view(-1, *inputs.v.shape[1:])[:token_offset].clone()
    if pending is not None:
        even, group_state = pending
        output[even] = _output_even(padded, corrected_u[even], group_state, even)
    return StreamingOutputs(
        output.view(-1, *inputs.v.shape[1:])[: inputs.length],
        state,
        corrected_u,
        boundary_states,
        boundary_outputs,
    )


def adaptive_partition_policy(
    inputs: GDNInputs,
    piece_lengths: Sequence[int],
    solve: str = "block-adjacent",
) -> tuple[StreamingOutputs, torch.Tensor, torch.Tensor]:
    if any(length % CHUNK_SIZE for length in piece_lengths):
        raise ValueError("adaptive partition policy currently requires 64-token alignment")
    masks = []
    tiers = []
    offset = 0
    for piece_length in piece_lengths:
        piece = slice_inputs(inputs, offset, offset + piece_length)
        reference = run_reference(piece, solve=solve, selection="adaptive")
        masks.append(reference.selector.mask)
        tiers.append(reference.selector.tiers)
        offset += piece_length
    mask = torch.cat(masks, dim=1)
    tier_values = torch.stack(tiers)
    return (
        run_frozen_partition(inputs, piece_lengths, mask, tier_values[0], solve),
        mask,
        tier_values,
    )


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    difference = actual.double() - expected.double()
    reference_norm = expected.double().square().sum().sqrt().item()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_frobenius": difference.square().sum().sqrt().item() / max(reference_norm, 1e-300),
        "torch_equal": torch.equal(actual, expected),
    }
