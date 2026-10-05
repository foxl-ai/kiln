"""Padded rows never write the cache slot they read (engine/model_runner.py ModelRunner.pad_slots).

A padded row of a static-shape call sits at position 0 behind an all-null block table, so the one key it
can see is slot 0 of the null page. Padded rows used to write their own KV there as well: under DP
attention every padded row of a group wrote that one slot and read it back in the same graph, and on
trn1 the value read back changed from execution to execution of GLM-5.3-Flash's 12-layer decode graphs
with identical inputs (docs/neuron-notes.md "Padded rows wrote the slot they read"; the device check
is tools/check_cold_determinism.py RD@k:n --assert-stable). The host cannot show that race (CPU
index_put_ is sequential), so this checks the invariant that rules it out: after warmup and serving
every call form with padding (padded decode rows, a DP group with no work, a chunk shorter than its
bucket, padded verify drafts, mixed batches), slot 0 of the null page is still the zeros it was
created with in every paged cache, while padding did write the null page's other slots.
"""

import pytest
import torch

from tests.test_architectures import build as build_arch
from tests.test_attention_tp import prompts


def null_page_slots(eng):
    """Rank 0's paged caches: (slot 0 of the null page, the rest of the null page) of each."""
    r = eng.runner
    return [(c[0].clone(), c[1 : r.ps].clone()) for c in r._paged_caches()]


def serve(path, ps, sp, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    # One 8-token chunk per DP-attention group and step: the prefill bucket below.
    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=8, num_pages=128, max_num_seqs=4,
                max_model_len=256, max_prefill_tokens=8 * kw.get("dp_attention", 1))
    base.update(kw)
    eng = LLMEngine(EngineConfig(**base))
    try:
        eng.warmup()
        eng.generate(ps, sp)
        return null_page_slots(eng)
    finally:
        eng.close()


def check(slots):
    assert slots, "no paged caches"
    for i, (zero, rest) in enumerate(slots):
        assert torch.count_nonzero(zero) == 0, f"cache {i}: a padded row wrote slot 0 of the null page"
    assert any(torch.count_nonzero(rest) for _, rest in slots), "no padded row wrote the null page at all"


@pytest.mark.parametrize("kw", [dict(), dict(tp=2, dp_attention=2), dict(tp=2, dp_attention=2, overlap=True),
                                dict(mixed_batch=True), dict(spec_method="ngram", spec_k=3)],
                         ids=["tp1", "dp2", "dp2-overlap", "mixed", "verify"])
def test_qwen3_padded_rows_never_write_slot_0(tmp_path, kw):
    from kiln.engine.request import SamplingParams

    build_arch("qwen3", str(tmp_path))
    # Uneven prompts: chunks shorter than the 8-token buckets, a decode batch below its bucket, and (DP
    # attention) groups that run padding while the other prefills.
    ps = prompts(3, (11, 5, 19))
    rep = [p * 4 for p in prompts(4, (3,))]  # n-gram drafts on a repeated pattern (verify rows padded)
    sp = SamplingParams(max_new_tokens=9, ignore_eos=True)
    check(serve(str(tmp_path), ps + (rep if kw.get("spec_method") else []), sp,
                prefill_token_buckets=(8,), decode_batch_buckets=(4,), **kw))


def test_glm5_next_padded_rows_never_write_slot_0(tmp_path):
    """GLM-5.3-Flash truncated (KDA, pooled DSA with in-place pool keys: the indexer rows are written twice
    per call through slot_mapping) under DP attention, where the device race was measured."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build

    build(str(tmp_path), index_topk=16)
    ps = prompts(8, (11, 41, 26))
    sp = SamplingParams(max_new_tokens=9, ignore_eos=True)
    check(serve(str(tmp_path), ps, sp, tp=2, dp_attention=2, overlap=True, prefill_token_buckets=(8,),
                decode_batch_buckets=(4,), max_num_seqs=3))
