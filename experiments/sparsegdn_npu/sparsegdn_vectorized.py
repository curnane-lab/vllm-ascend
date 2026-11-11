from __future__ import annotations

import math

import torch
from sparsegdn_reference import (
    CHUNK_SIZE,
    TAIL_SIZE,
    GDNInputs,
    PaddedInputs,
    ReferenceOutputs,
    _value_head_keys,
    frozen_selection,
    full_selection,
    pad_inputs,
    select_events,
)


def _chunked_inputs(inputs: GDNInputs) -> PaddedInputs:
    padded = PaddedInputs(**vars(pad_inputs(inputs)))
    for name in ("q", "k", "v", "log_decay", "beta", "z"):
        value = getattr(padded, name)
        setattr(
            padded,
            name,
            value.view(padded.num_chunks, CHUNK_SIZE, *value.shape[1:]),
        )
    return padded


def _delta_matrices(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    interactions = k @ k.transpose(-1, -2)
    lower = torch.tril(torch.ones(64, 64, dtype=torch.bool, device=k.device), diagonal=-1)
    exponent = torch.where(
        lower,
        g[:, :, :, None] - g[:, :, None, :],
        torch.zeros_like(interactions),
    )
    decay = torch.exp2(exponent)
    identity = torch.eye(64, dtype=k.dtype, device=k.device)
    return identity + beta[:, :, :, None] * interactions * decay * lower


def _compute_intra(padded: PaddedInputs, solve: str):
    chunks, _, heads, value_dim = padded.v.shape
    k = _value_head_keys(padded.k, heads).transpose(1, 2)
    v = padded.v.transpose(1, 2)
    g = padded.g.transpose(1, 2)
    beta = padded.beta.transpose(1, 2)
    matrix = _delta_matrices(k, g, beta)
    source_scale = beta * torch.exp2(g)
    value_rhs = beta[:, :, :, None] * v
    if solve == "full64":
        coefficients = torch.linalg.inv(matrix)
        u = coefficients @ value_rhs
        read_coefficients = coefficients * source_scale[:, :, None, :]
    elif solve == "block-adjacent":
        blocks = matrix.view(chunks, heads, 4, 16, 4, 16).permute(0, 1, 2, 4, 3, 5)
        indices = torch.arange(4, device=matrix.device)
        diagonal = torch.linalg.inv(blocks[:, :, indices, indices])
        adjacent = -diagonal[:, :, 1:] @ blocks[:, :, indices[1:], indices[:-1]] @ diagonal[:, :, :-1]
        value_blocks = value_rhs.view(chunks, heads, 4, 16, value_dim)
        u_blocks = diagonal @ value_blocks
        u_blocks[:, :, 1:] += adjacent @ value_blocks[:, :, :-1]
        u = u_blocks.reshape(chunks, heads, 64, value_dim)
        diagonal_scale = source_scale.view(chunks, heads, 4, 16)
        diagonal_read = diagonal * diagonal_scale[:, :, :, None, :]
        adjacent_read = adjacent * diagonal_scale[:, :, :-1, None, :]
        read_coefficients = torch.cat((diagonal_read, adjacent_read), dim=2)
    else:
        raise ValueError(f"unknown solve: {solve}")
    u = u.transpose(1, 2)
    scores = u[:, -TAIL_SIZE:].square().sum(dim=(1, 3)).sqrt()
    return u, read_coefficients, scores


def _event_step(
    padded: PaddedInputs,
    solve: str,
    read_coefficients: torch.Tensor,
    u: torch.Tensor,
    mask: torch.Tensor,
    state: torch.Tensor,
    chunk: int,
):
    heads = padded.v.shape[-2]
    k = _value_head_keys(padded.k, heads)
    selected = mask[:, chunk].repeat_interleave(2).nonzero().flatten()
    w = torch.zeros(
        64,
        heads,
        padded.k.shape[-1],
        dtype=u.dtype,
        device=u.device,
    )
    if selected.numel():
        selected_k = k[chunk][:, selected]
        coefficients = read_coefficients[chunk, selected]
        if solve == "full64":
            selected_w = torch.einsum("stl,lsk->stk", coefficients, selected_k)
            w[:, selected] = selected_w.permute(1, 0, 2)
        else:
            key_blocks = selected_k.view(4, 16, selected.numel(), -1).permute(2, 0, 1, 3)
            w_blocks = coefficients[:, :4] @ key_blocks
            w_blocks[:, 1:] += coefficients[:, 4:] @ key_blocks[:, :-1]
            w[:, selected] = w_blocks.reshape(selected.numel(), 64, -1).permute(1, 0, 2)
    corrected = u[chunk] - torch.einsum("lhk,hkv->lhv", w, state)
    weights = torch.exp2(padded.gamma[chunk][None] - padded.g[chunk])[:, :, None]
    head_mask = mask[:, chunk].repeat_interleave(2)
    writes = torch.einsum("lhk,lhv->hkv", k[chunk], corrected * weights)
    writes = writes * head_mask[:, None, None]
    state = torch.exp2(padded.gamma[chunk])[:, None, None] * state + writes
    return corrected, state


def _output_even(
    padded: PaddedInputs,
    corrected: torch.Tensor,
    entry_state: torch.Tensor,
    chunk: int,
):
    heads = padded.v.shape[-2]
    q = _value_head_keys(padded.q, heads)[chunk].transpose(0, 1)
    k = _value_head_keys(padded.k, heads)[chunk].transpose(0, 1)
    g = padded.g[chunk].transpose(0, 1)
    u = corrected.transpose(0, 1)
    causal = torch.tril(torch.ones(64, 64, dtype=torch.bool, device=q.device))
    exponent = torch.where(
        causal,
        g[:, :, None] - g[:, None, :],
        torch.zeros(64, 64, dtype=g.dtype, device=g.device),
    )
    local = (q @ k.transpose(-1, -2)) * torch.exp2(exponent) * causal
    historical = torch.exp2(g[:, :, None]) * (q @ entry_state)
    return ((historical + local @ u) / math.sqrt(padded.k.shape[-1])).transpose(0, 1)


def _output_odd(
    padded: PaddedInputs,
    even_u: torch.Tensor,
    odd_u: torch.Tensor,
    entry_state: torch.Tensor,
    even: int,
):
    odd = even + 1
    heads = padded.v.shape[-2]
    q = _value_head_keys(padded.q, heads)[odd].transpose(0, 1)
    even_k = _value_head_keys(padded.k, heads)[even].transpose(0, 1)
    odd_k = _value_head_keys(padded.k, heads)[odd].transpose(0, 1)
    odd_g = padded.g[odd].transpose(0, 1)
    even_g = padded.g[even].transpose(0, 1)
    causal = torch.tril(torch.ones(64, 64, dtype=torch.bool, device=q.device))
    own_exponent = torch.where(
        causal,
        odd_g[:, :, None] - odd_g[:, None, :],
        torch.zeros(64, 64, dtype=odd_g.dtype, device=odd_g.device),
    )
    own = (q @ odd_k.transpose(-1, -2)) * torch.exp2(own_exponent) * causal
    cross_decay = odd_g[:, :, None] + padded.gamma[even][:, None, None]
    cross = (q @ even_k.transpose(-1, -2)) * torch.exp2(cross_decay - even_g[:, None, :])
    historical = torch.exp2(cross_decay) * (q @ entry_state)
    result = historical + cross @ even_u.transpose(0, 1) + own @ odd_u.transpose(0, 1)
    return (result / math.sqrt(padded.k.shape[-1])).transpose(0, 1)


def run_vectorized_reference(
    inputs: GDNInputs,
    solve: str = "block-adjacent",
    selection: str = "adaptive",
    frozen_mask: torch.Tensor | None = None,
    frozen_tiers: torch.Tensor | None = None,
    return_state_trace: bool = False,
) -> ReferenceOutputs:
    padded = _chunked_inputs(inputs)
    u, read_coefficients, scores = _compute_intra(padded, solve)
    from sparsegdn_reference import IntraOutputs

    intra = IntraOutputs(u, read_coefficients, scores, solve)
    if frozen_mask is not None:
        selector = frozen_selection(frozen_mask, frozen_tiers, padded, intra)
    elif selection == "adaptive":
        selector = select_events(inputs, padded, intra)
    elif selection == "full":
        selector = full_selection(padded, intra)
    else:
        raise ValueError(f"unknown selection: {selection}")
    corrected_u = torch.empty_like(u)
    state = padded.initial_state.clone()
    trace = [state.clone()]
    for chunk in range(padded.num_chunks):
        corrected_u[chunk], state = _event_step(
            padded,
            solve,
            read_coefficients,
            u,
            selector.mask,
            state,
            chunk,
        )
        if return_state_trace:
            trace.append(state.clone())
    if not return_state_trace:
        raise ValueError("the vectorized driver requires return_state_trace=True")
    output = torch.full_like(u, float("nan"))
    for even in range(0, padded.num_chunks, 2):
        output[even] = _output_even(padded, corrected_u[even], trace[even], even)
        if even + 1 < padded.num_chunks:
            output[even + 1] = _output_odd(
                padded,
                corrected_u[even],
                corrected_u[even + 1],
                trace[even],
                even,
            )
    return ReferenceOutputs(
        output.reshape(-1, *inputs.v.shape[1:])[: inputs.length],
        state,
        torch.stack(trace),
        corrected_u,
        intra,
        selector,
    )
