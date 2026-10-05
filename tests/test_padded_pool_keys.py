"""Padded rows never write a pool key a real request owns (models/mla.py _pool_key_slots).

With the separate pool-key cache (KILN_DSA_POOL_CACHE=separate, the default under an FP8 KV cache:
GLM-5.3-Flash's G64 serving configuration), write_pool_keys recomputes the key of every written
token's pool through the block table. A padded row sits at position 0, so in a chunk's padded tail
(the table is the real sequence's) its pool was the real sequence's pool 0: padded rows rewrote that
pool's key slots, beside the real rows that own them, from whatever its rows held when they read them.
Identical values only if both read the same rows, which on the device is the read-after-write order
that failed for the null page (docs/neuron-notes.md "Padded rows wrote the slot they read"). Now a
padded row writes only its own null-page slot.

test_padded_tail_does_not_write_the_sequence_pool_0 gives the padded rows different source data from
the key the sequence holds and checks that key is untouched; the engine case checks the invariant of
tests/test_padded_rows.py (nothing writes the null page's slot 0, here including the pool-key cache)
under DP attention.
"""

import inspect

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

from tests.test_attention_tp import prompts  # noqa: E402


@pytest.fixture
def separate(monkeypatch):
    """The separate pool-key cache in this process (mla.POOL_CACHE is read at import) and in tensor-parallel
    workers, which import it afresh from the environment."""
    from kiln.models import mla

    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")


def engine(path, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=8, num_pages=64, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=8 * kw.get("dp_attention", 1), prefill_token_buckets=(8,),
                decode_batch_buckets=(4,))
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def test_padded_tail_does_not_write_the_sequence_pool_0(tmp_path, separate):
    """A chunk of 4 real tokens (positions 8..11, pool 2) and 2 padded rows (position 0) over a sequence's
    table: pool 0's key slots hold a key (a sentinel here) and its rows hold OTHER data, so a padded row
    recomputing pool 0 from them would change the key. Red on the tree where write_pool_keys took no
    slot_mapping: the padded rows overwrote the sentinel."""
    from kiln.models import mla
    from kiln.models.glm5_next import pool_keys
    from tests.test_glm5_next import build

    build(str(tmp_path), index_topk=16)
    eng = engine(str(tmp_path))
    try:
        model = eng.model
        layer = next(l for l in model.layers if getattr(l, "pool_key", None) is not None)
        d = layer.spec.mla.dsa
        ps, kp = model.page_size, d.kpool
        p0, p1 = 5, 6  # the sequence's pages: positions 0..7 and 8..15
        table = torch.tensor([p0, p1, 0, 0])
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            vc, pk = layer.v_cache, layer.pool_key
            vc[p0 * ps : p0 * ps + kp] = torch.randn(vc[:kp].shape, generator=g)  # pool 0's rows: other data
            vc[p1 * ps : p1 * ps + kp] = torch.randn(vc[:kp].shape, generator=g)  # the chunk's rows (stored)
            pk.fill_(0.0)
            pk[p0 * ps : p0 * ps + kp] = 123.0  # the key pool 0 holds
            positions = torch.tensor([8, 9, 10, 11, 0, 0])
            slots = torch.tensor([p1 * ps + i for i in range(4)] + [1, 2])  # padded rows: null-page slots 1, 2
            args = (model, layer, d, positions, table)
            if "slot_mapping" in inspect.signature(mla.write_pool_keys).parameters:
                mla.write_pool_keys(*args, slots)
            else:
                mla.write_pool_keys(*args)
        assert torch.all(pk[p0 * ps : p0 * ps + kp] == 123.0), "a padded row rewrote the sequence's pool 0 key"
        Di, dr = d.head_dim, layer.spec.mla.qk_rope_head_dim
        want = pool_keys(layer, d, vc[p1 * ps : p1 * ps + kp, 0, dr : dr + 2 * Di].unsqueeze(0))[0]
        got = pk[p1 * ps : p1 * ps + kp].reshape(-1)
        err = (got - want.to(pk.dtype)).abs().max().item()
        assert err < 1e-5, f"the real rows' pool 2 key (max |d| {err})"  # fp32: the same arithmetic over 6 rows
        assert torch.count_nonzero(pk[0]) == 0, "a padded row wrote the null page's slot 0"
    finally:
        eng.close()


def test_padded_rows_never_write_slot_0_with_separate_pool_keys(tmp_path, separate):
    """tests/test_padded_rows.py's invariant with the separate pool-key cache (decode rows padded under DP
    attention, chunks shorter than their bucket): slot 0 of the null page of every paged cache, the
    pool-key cache included, is still zero after serving."""
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build
    from tests.test_padded_rows import check, null_page_slots

    build(str(tmp_path), index_topk=16)
    eng = engine(str(tmp_path), tp=2, dp_attention=2, overlap=True, num_pages=128)
    try:
        assert eng.runner.s_caches, "no separate pool-key cache"
        eng.warmup()
        eng.generate(prompts(8, (11, 41, 26)), SamplingParams(max_new_tokens=9, ignore_eos=True))
        check(null_page_slots(eng))
    finally:
        eng.close()
