import torch

from kiln.engine.sampler import NUM_TOP_LOGPROBS, sample, unpack


def run(logits, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, K=64, seed=0):
    B = logits.shape[0]
    g = torch.Generator().manual_seed(seed)
    f = lambda v: torch.full((B,), v, dtype=torch.float32)  # noqa: E731
    return sample(logits, f(temperature), f(top_p), torch.full((B,), top_k, dtype=torch.int64),
                  f(min_p), torch.rand(B, K, generator=g))


def test_greedy_and_raw_logprobs():
    logits = torch.randn(3, 1000)
    tokens, lps, top_ids, top_lps = unpack(run(logits), 3)
    ref = torch.log_softmax(logits, -1)
    assert tokens == logits.argmax(-1).tolist()
    for b in range(3):
        assert abs(lps[b] - ref[b, tokens[b]].item()) < 1e-4
        assert top_ids[b] == torch.topk(logits[b], NUM_TOP_LOGPROBS).indices.tolist()
        assert abs(top_lps[b][0] - ref[b].max().item()) < 1e-4


def test_top_k_one_is_greedy_even_with_temperature():
    logits = torch.randn(4, 500)
    tokens, *_ = unpack(run(logits, temperature=1.5, top_k=1), 4)
    assert tokens == logits.argmax(-1).tolist()


def test_min_p_drops_tokens_below_threshold():
    logits = torch.full((1, 100), -10.0)
    logits[0, 5], logits[0, 6], logits[0, 7] = 5.0, 4.9, 0.0  # p(7) << 0.5 * p(5)
    seen = set()
    for seed in range(200):
        tokens, *_ = unpack(run(logits, temperature=1.0, min_p=0.5, seed=seed), 1)
        seen.add(tokens[0])
    assert seen == {5, 6}


def test_sampling_matches_the_distribution():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2] + [1e-9] * 61]))
    counts = torch.zeros(64)
    n = 4000
    for seed in range(n // 50):
        out = run(logits.repeat(50, 1), temperature=1.0, seed=seed)
        tokens, *_ = unpack(out, 50)
        for t in tokens:
            counts[t] += 1
    freq = counts[:3] / n
    assert torch.allclose(freq, torch.tensor([0.5, 0.3, 0.2]), atol=0.03), freq


def test_top_p_keeps_the_smallest_set_covering_p():
    logits = torch.log(torch.tensor([[0.6, 0.25, 0.1, 0.05] + [1e-9] * 60]))
    seen = set()
    for seed in range(300):
        tokens, *_ = unpack(run(logits, temperature=1.0, top_p=0.8, seed=seed), 1)
        seen.add(tokens[0])
    assert seen == {0, 1}


def test_penalties_match_vllm_semantics():
    from kiln.engine.sampler import apply_penalties

    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0, 1.0]])
    hist = torch.tensor([[0, 1, 3, 0]])  # padded by repeating the first id
    counts = torch.tensor([[2.0, 0.0, 1.0, 2.0]])  # token 1 seen in the prompt only
    out = apply_penalties(logits, hist, counts, torch.tensor([2.0]), torch.tensor([0.1]), torch.tensor([0.5]))
    expect = torch.tensor([[2.0 / 2 - 0.2 - 0.5, -1.0 * 2, 0.5, 3.0 / 2 - 0.1 - 0.5, 1.0]])
    assert torch.allclose(out, expect)


def test_penalized_token_is_not_sampled_but_logprobs_stay_raw():
    logits = torch.tensor([[5.0, 4.0] + [-10.0] * 62])
    B = 1
    f = lambda v: torch.full((B,), v)  # noqa: E731
    pen = (torch.tensor([[0]]), torch.tensor([[3.0]]), f(1.0), f(1.0), f(0.0))
    out = sample(logits, f(0.0), f(1.0), torch.zeros(B, dtype=torch.int64), f(0.0), torch.rand(B, 64),
                 penalties=pen)
    tokens, lps, top_ids, _ = unpack(out, 1)
    assert tokens == [1] and top_ids[0][0] == 0
    assert abs(lps[0] - torch.log_softmax(logits, -1)[0, 1].item()) < 1e-5
