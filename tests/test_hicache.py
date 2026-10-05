"""Host KV tier (engine/hicache.py): evicted prefixes come back from host memory."""

import torch


def _engine(path, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    return LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=24,
                                  max_num_seqs=1, max_model_len=96, max_prefill_tokens=64, **kw))


import pytest


@pytest.mark.parametrize("tp", [1, 2])
def test_evicted_prefix_is_restored_from_host_and_output_is_exact(tmp_path, tp):
    from tests.test_architectures import build

    from kiln.engine.request import SamplingParams

    build("qwen3", str(tmp_path))
    sp = SamplingParams(max_new_tokens=6, ignore_eos=True)
    convo = list(range(100, 164))  # 16 pages
    churn = [[(7 * i + j) % 383 for j in range(60)] for i in range(3)]

    def run(eng):
        first = eng.generate([convo], sp)[0].output_ids
        for c in churn:  # a 24-page pool cannot keep the conversation through these
            eng.generate([c], sp)
        (again,) = eng.generate([convo + first], sp)
        return first, again

    plain_first, plain_again = run(_engine(str(tmp_path)))
    eng = _engine(str(tmp_path), hicache_host_gb=0.01, tp=tp)
    try:
        first, again = run(eng)
    finally:
        eng.close()
    assert (first, again.output_ids) == (plain_first, plain_again.output_ids)
    assert eng.host_tier.saved > 0 and eng.host_tier.restored >= 16
    assert again.num_cached_tokens >= 64  # the whole conversation, back from the host
    assert plain_again.num_cached_tokens < 64


def test_host_tier_lru_and_pinning():
    from kiln.engine.hicache import HostTier

    class Moves:
        def __init__(self):
            self.log = []

        def kv_save(self, slot, page):
            self.log.append(("save", slot, page))

        def kv_load(self, slot, page):
            self.log.append(("load", slot, page))

        def kv_drop(self, slot):
            self.log.append(("drop", slot))

    mv = Moves()
    t = HostTier(mv, page_size=2, capacity_pages=2)
    t.offload([1, 2, 3, 4], 0, [10, 11])  # two pages of one path
    assert t.lookup([1, 2, 3, 4, 5], 0, 2) == [0, 1] and t.lookup([1, 2, 9, 9], 0, 2) == [0]
    t.unpin([0, 1])
    t.lookup([1, 2], 0, 1)  # pins the first page
    t.offload([7, 7], 0, [12])  # full: drops the LRU page that is not pinned
    assert ("drop", 1) in mv.log and t.lookup([1, 2], 0, 1) == [0]
