# Debug Session: dflash-k9-acceptance

Status: OPEN

## Symptom

DFlash on vllm-ascend remains at ~0.2%-0.5% acceptance for K=9 after applying:

- RGDR MAX_MTP expansion
- targeted #8906 DFlash acceptance fixes
- GDN spec metadata slicing
- arch35 MAX_MTP=16

Observed metrics show pos0 acceptance around 2%-4% and positions 1..8 near 0%.

## Current constraints

- Do not apply further business-logic fixes before collecting runtime evidence.
- The current pending commit is instrumentation-only and is gated by `DFLASH_K9_DEBUG=1`.

## Hypotheses

1. Draft logits are produced over the wrong vocabulary or d2t remapping is not applied before verification.
   - Evidence needed: draft vocab size, presence of draft_id_to_target_id, sampled draft ids before/after remap.

2. DFlash query construction is correct for K=7 but wrong for K=9, causing sample rows to point to wrong hidden states.
   - Evidence needed: input_ids, positions, slot_mapping, token_indices_to_sample for one batch.

3. Target auxiliary hidden states used by DFlash are wrong for Qwen3.5, either wrong layers or wrong tensor layout.
   - Evidence needed: configured target_layer_ids, actual aux layer ids, aux_hidden_states shape/type.

4. Qwen3.5 GDN/attention spec path still consumes stale or padded metadata despite the previous slicing patch.
   - Evidence needed: num_spec_decodes, spec_query_start_loc, spec_state_indices, num_accepted_tokens, actual_seq_lengths.

5. The runtime is not actually loading the rebuilt custom op or latest branch code.
   - Evidence needed: startup git commit/build marker and explicit `DFLASH_K9_DEBUG` lines from patched code paths.

## Instrumentation points

The debug commit emits one-time server log lines containing the marker `DFLASH_K9_DEBUG` from:

- DFlash first pass query expansion
- DFlash sample row selection and logits/d2t remapping
- DFlash non-reduce sampling raw argmax and optional d2t remap
- Model runner aux hidden state handoff
- GDN spec decode metadata

## How to collect logs

Run the server with `DFLASH_K9_DEBUG=1`, reproduce single-instance K=9, then collect:

```bash
grep -n "DFLASH_K9_DEBUG\|SpecDecoding metrics" <server_log> > /tmp/dflash_k9_debug.log
```

Paste `/tmp/dflash_k9_debug.log` back into the conversation.
