"""On-device sampling, compiled into the same graph as the model.

Greedy, temperature, top-k, top-p and min-p over a fixed candidate set of the K highest
logits (K = noise.shape[1]). Exact for top_k <= K; for a wider distribution the mass
outside the top K is dropped, which is the price of a static shape. Randomness comes in
as uniform noise generated on the host per request, so a seeded request is reproducible
and the graph has no RNG state. Rows with temperature <= 0 are greedy over the full vocab.

Logprobs follow vLLM's default (`logprobs_mode=raw_logprobs`): log-softmax of the raw
model logits, before temperature and filtering.

The graph returns ONE fp32 tensor per call, so a step is a single device-to-host read:
    out[:, 0]          sampled token id (exact in fp32 for ids < 2**24)
    out[:, 1]          its logprob
    out[:, 2:2+N]      top-N token ids
    out[:, 2+N:2+2N]   top-N logprobs
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_TOP_LOGPROBS = 20  # the OpenAI API maximum for top_logprobs


def topk_large(x: torch.Tensor, k: int, chunk: int = 128):
    """Exact top-k over a long last axis, for [B small, V large] on Trainium.

    A plain topk / argmax / logsumexp over [B, V] puts B rows on 128 partitions, so a decode
    batch uses a few lanes (measured on trn1: topk(64) over Qwen3's 151,936-way vocab is
    3.37 ms at B=6). Folding to [B, V / chunk, chunk] makes the rows plentiful: take each
    chunk's max, the k best chunks, then top-k inside those k * chunk candidates. Exact,
    because every element of the true top-k lies in a chunk whose max is among the top k.
    """
    B, V = x.shape
    if -(-V // chunk) < k:  # fewer chunks than k: nothing to gain
        return torch.topk(x, k, dim=-1)
    pad = (-V) % chunk
    if pad:
        x = F.pad(x, (0, pad), value=-1e30)
    xc = x.view(B, -1, chunk)
    cmax = xc.amax(dim=-1)
    _, ci = torch.topk(cmax, k, dim=-1)
    cand = torch.gather(xc, 1, ci.unsqueeze(-1).expand(B, k, chunk)).reshape(B, k * chunk)
    vals, j = torch.topk(cand, k, dim=-1)
    idx = torch.gather(ci, 1, torch.div(j, chunk, rounding_mode="floor")) * chunk + j % chunk
    return vals, idx


def logsumexp_large(x: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """logsumexp over a long last axis, folded the same way. Returns [B, 1]."""
    B, V = x.shape
    if V <= chunk:
        return torch.logsumexp(x, dim=-1, keepdim=True)
    pad = (-V) % chunk
    if pad:
        x = F.pad(x, (0, pad), value=-1e30)
    return torch.logsumexp(torch.logsumexp(x.view(B, -1, chunk), dim=-1), dim=-1, keepdim=True)


def unpack_bitmask(bitmask: torch.Tensor, vocab: int) -> torch.Tensor:
    """xgrammar's packed int32 bitmask [B, ceil(V/32)] -> bool [B, V] (True = allowed)."""
    shifts = torch.arange(32, dtype=torch.int32, device=bitmask.device)
    bits = torch.bitwise_and(torch.bitwise_right_shift(bitmask.unsqueeze(-1), shifts), 1)
    return bits.reshape(bitmask.shape[0], -1)[:, :vocab] != 0


def apply_penalties(logits: torch.Tensor, hist_ids: torch.Tensor, out_counts: torch.Tensor,
                    rep: torch.Tensor, freq: torch.Tensor, pres: torch.Tensor) -> torch.Tensor:
    """vLLM semantics. hist_ids [B, M]: every distinct token in the prompt or the output
    (rows padded by repeating their first entry, which makes the scatter idempotent);
    out_counts [B, M]: occurrences in the output only. repetition_penalty divides positive
    and multiplies negative logits of any seen token; frequency and presence penalties
    subtract per output occurrence and once per distinct output token."""
    g = torch.gather(logits, 1, hist_ids)
    r = rep.unsqueeze(1)
    g = torch.where(g > 0, g / r, g * r)
    g = g - freq.unsqueeze(1) * out_counts - pres.unsqueeze(1) * (out_counts > 0).float()
    return torch.scatter(logits, 1, hist_ids, g)


def sample(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor,
           top_k: torch.Tensor, min_p: torch.Tensor, noise: torch.Tensor,
           bitmask: torch.Tensor | None = None, penalties: tuple | None = None) -> torch.Tensor:
    """logits [B, V] fp32; temperature/top_p/min_p [B] fp32; top_k [B] int64 (0 = off);
    noise [B, K] uniform in (0, 1); bitmask: optional grammar mask (see grammar.py);
    penalties: optional (hist_ids, out_counts, rep, freq, pres) for apply_penalties. Both
    shape what is sampled, never the reported raw logprobs. Returns [B, 2 + 2N] fp32."""
    K = noise.shape[1]
    N = NUM_TOP_LOGPROBS
    lse = logsumexp_large(logits)
    raw_vals, raw_idx = topk_large(logits, max(K, N))  # sorted descending
    if bitmask is None and penalties is None:
        vals, idx = raw_vals, raw_idx
    else:
        slog = logits if penalties is None else apply_penalties(logits, *penalties)
        if bitmask is not None:
            slog = torch.where(unpack_bitmask(bitmask, logits.shape[1]), slog, -1e30)
        vals, idx = topk_large(slog, max(K, 1))
    greedy = idx[:, 0]
    if K > 1:
        cv, ci = vals[:, :K], idx[:, :K]
        t = torch.clamp(temperature, min=1e-5).unsqueeze(1)
        probs = torch.softmax(cv / t, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        rank = torch.arange(K, device=logits.device).unsqueeze(0)
        keep = (cum - probs) < top_p.unsqueeze(1)
        keep = keep & ((rank < top_k.unsqueeze(1)) | (top_k.unsqueeze(1) <= 0))
        keep = keep & (probs >= min_p.unsqueeze(1) * probs[:, :1])
        keep = keep | (rank == 0)  # never an empty set
        gumbel = -torch.log(-torch.log(torch.clamp(noise, 1e-10, 1.0 - 1e-7)))
        scores = torch.where(keep, torch.log(probs) + gumbel, -1e30)
        choice = torch.argmax(scores, dim=-1, keepdim=True)
        sampled = torch.gather(ci, 1, choice).squeeze(1)
        token = torch.where(temperature <= 0, greedy, sampled)
    else:
        token = greedy
    token_lp = torch.gather(logits, 1, token.unsqueeze(1)) - lse
    top_lp = raw_vals[:, :N] - lse
    return torch.cat([token.unsqueeze(1).float(), token_lp, raw_idx[:, :N].float(), top_lp], dim=1)


def score_rows(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Raw scores of given tokens (prompt logprobs): logits [R, V] fp32, targets [R] int64.
    Returns [R, 2 + 2N]: rank of the target (1 = argmax; vLLM's rank counts strictly larger
    logits, +1), its logprob, then the top-N ids and logprobs, the same layout as sample()."""
    N = NUM_TOP_LOGPROBS
    lse = logsumexp_large(logits)
    tgt = torch.gather(logits, 1, targets.unsqueeze(1))
    rank = (logits > tgt).sum(dim=1, keepdim=True).float() + 1
    vals, idx = topk_large(logits, N)
    return torch.cat([rank, tgt - lse, idx[:, :N].float(), vals[:, :N] - lse], dim=1)


VERIFY_COLS = 6 + 2 * NUM_TOP_LOGPROBS


def verify_sample(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor,
                  top_k: torch.Tensor, min_p: torch.Tensor, noise: torch.Tensor,
                  u_accept: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """Speculative verification for one draft token per row (R = B x Q rows).

    Row i scores the draft for the NEXT position, d = draft[i], under the request's own
    sampling distribution p (temperature, top-k, top-p, min-p over the K candidates):
    accept with probability p(d) (greedy rows: d == argmax), else the replacement is a
    sample from p with d removed. For a deterministic proposal such as an n-gram this is
    exactly the rejection sampler vLLM uses, so the output distribution is unchanged.

    Columns: y (sample from p, the bonus token after a fully accepted draft), accept,
    r (replacement), raw logprobs of y, r and d, then top-N ids and logprobs.
    """
    R = logits.shape[0]
    K = noise.shape[1]
    N = NUM_TOP_LOGPROBS
    lse = logsumexp_large(logits)
    vals, idx = topk_large(logits, max(K, N))
    cv, ci = vals[:, :K], idx[:, :K]
    t = torch.clamp(temperature, min=1e-5).unsqueeze(1)
    probs = torch.softmax(cv / t, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    rank = torch.arange(K, device=logits.device).unsqueeze(0)
    keep = (cum - probs) < top_p.unsqueeze(1)
    keep = keep & ((rank < top_k.unsqueeze(1)) | (top_k.unsqueeze(1) <= 0))
    keep = keep & (probs >= min_p.unsqueeze(1) * probs[:, :1])
    keep = keep | (rank == 0)
    kept = torch.where(keep, probs, 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    greedy = ci[:, 0]
    is_d = ci == draft.unsqueeze(1)
    p_d = (kept * is_d.float()).sum(dim=-1)
    sampling = temperature > 0
    accept = torch.where(sampling, u_accept < p_d, draft == greedy)
    gumbel = -torch.log(-torch.log(torch.clamp(noise, 1e-10, 1.0 - 1e-7)))
    scores = torch.where(keep, torch.log(kept) + gumbel, -1e30)
    y = torch.where(sampling, torch.gather(ci, 1, torch.argmax(scores, -1, keepdim=True)).squeeze(1), greedy)
    rscores = torch.where(is_d, -1e30, scores)
    rsampled = torch.gather(ci, 1, torch.argmax(rscores, -1, keepdim=True)).squeeze(1)
    greedy_r = torch.where(draft == greedy, ci[:, 1], greedy)
    r = torch.where(sampling, rsampled, greedy_r)
    lp = lambda tok: torch.gather(logits, 1, tok.unsqueeze(1)) - lse  # noqa: E731
    return torch.cat([y.unsqueeze(1).float(), accept.unsqueeze(1).float(), r.unsqueeze(1).float(),
                      lp(y), lp(r), lp(draft), idx[:, :N].float(), vals[:, :N] - lse], dim=1)


def unpack(out, n: int):
    """Host side: the first `n` rows of a sampler output as (tokens, logprobs, top_ids, top_lps)."""
    out = out[:n]
    N = NUM_TOP_LOGPROBS
    return (out[:, 0].long().tolist(), out[:, 1].tolist(),
            out[:, 2:2 + N].long().tolist(), out[:, 2 + N:2 + 2 * N].tolist())
