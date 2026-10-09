"""Speculative decoding under overlap scheduling (asynchronous MTP drafting; EngineConfig.spec_async).

The synchronous speculative step reads every verify's accepted count back to the host before it can build the
next step: the next verify's positions, KV slots, input tokens and recurrent-state row, and the MTP pass's
positions, tokens and hidden-state row all depend on it. On GLM-5.3-Flash at tp=32 that host round trip is
~50 ms of a ~0.26 s speculative step (docs/neuron-notes.md "MTP speculative decoding on GLM-5.3-Flash at
serving scale": ~31 s of host time over 612 verify steps at G1b warm).

The reference engines keep that state on the device and schedule the next step blind:
- vLLM v0.30.0 v1/core/sched/async_scheduler.py:19-49 schedules every decode row with 1 + k tokens as if every
  draft is accepted; Model Runner V2 (worker/gpu/model_runner.py:1997-2181, worker/gpu/states.py:58-73) keeps
  num_computed_tokens, the last sampled tokens and the drafts on the GPU, corrects num_computed_tokens on the
  device after the rejection sampler, and builds the next step's input ids and positions from those buffers.
- SGLang v0.5.21 speculative/eagle_worker_v2.py:1313-1450 with managers/overlap_utils.py:513-594 publishes the
  new sequence lengths and bonus tokens into a device FutureMap that the next batch gathers from, reserving
  slots for both steps in flight (mem_cache/allocation_sizing.py:16-59).

Kiln's form: one fp32 board row per request slot (exact integers below 2^24) holds what the next speculative
step needs, and five small graphs around the UNCHANGED verify and MTP graphs (so their compile keys stay the
synchronous engine's) read and write it:

    columns 0 .. Q - 1   T: the tokens after positions base .. base + Q - 1 of the last verify, the first
                         acc + 1 meaningful (accepted drafts, then the replacement or the bonus token);
                         after a prefill T[0] is the sampled token
    column  Q            acc: drafts the last verify accepted (0 after a prefill)
    column  Q + 1        base: the position of the newest token, the next verify's first row
    column  Q + 2        cur: the recurrent-state row the next verify reads (linear-attention models)
    column  Q + 3        nd: whether the drafts below are real (k after an MTP pass, 0 before the first one: a
                         draftless row takes the sample y_0, as the synchronous engine's draftless verify rows do)
    columns Q + 4 ..     the k drafts of the last MTP pass

- spec_prep (before a verify): input ids (row 0 the newest token T[acc], rows 1 .. k the drafts), positions
  base .. base + k, KV slots through the host's page table, the state rows (cur, then the request's own rows),
  and the rejection sampler's draft column.
- spec_post (after it): the accepted count from the sampler's accept flags (the accepted prefix), the emitted
  tokens, base + 1 + acc, cur = the row after the last accepted position (the request's row acc), written into
  the board; a compact [rows, Q + 2] result (T, acc, nd) the host reads one step later.
- mtp_prep (after spec_post): the MTP pass's ids T, positions base_old .. base_old + k, KV slots and
  last_index = acc, so its draft comes from the last accepted position (positions past it hold tokens and KV
  that the next pass rewrites before anything reads them).
- mtp_post: the MTP drafts into the board.
- spec_init (after a request's final prefill chunk): T[0] from the token board the prefill wrote, acc 0, base
  and cur from the host; mtp_prefill_ids takes the chunk's MTP ids with the last one from the token board.
- spec_host_init (before a request's first blind verify when its newest token is a prompt token the host
  holds: a prompt whose chunks stopped one token short, or a recompute after a preemption): T[0] from the host.

Every rank runs every graph over all N * B rows (DP attention: the board is replicated and every rank computes
the same values); the per-group arguments a token mixer reads (positions, KV slots, state rows) come out for
the rank's own group through its dp_index buffer, as the host's PerGroup arguments would.

Positions are exact in fp32; the page of a position is floor(pos / page_size), exact for a power-of-two page
size (a division by a power of two only shifts the exponent), so no integer division enters a graph
(docs/neuron-notes.md: int64 x // c lowers through an fp32 reciprocal). No comparison is against a float literal
and no torch.where takes a float scalar: neuronx-cc 2.27 refused every board graph (seven shapes) written with
`valid > 0.5` and `torch.where(..., 0.0)` (NCC_ESPP004 "f64 dtype is not supported", compile farm 2026-10-05); the
0 / 1 flags compare against the integer 0, as the sampler's `temperature <= 0` does.
"""

from __future__ import annotations

import torch


def width(Q: int, k: int) -> int:
    return Q + 4 + k


def _onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    """fp32 [R, n] one-hot of exact-integer fp32 idx [R] (values outside 0 .. n - 1: all zero)."""
    ar = torch.arange(n, device=idx.device, dtype=torch.float32)
    return (idx.unsqueeze(1) == ar.unsqueeze(0)).to(torch.float32)


def _slots(pos: torch.Tensor, table: torch.Tensor, ps: int) -> torch.Tensor:
    """KV slots of fp32 positions [R, Q] through page tables [R, P] (int64): page floor(pos / ps), as a one-hot
    contraction over the P pages (exact: one nonzero term), then page * ps + pos mod ps."""
    pidx = torch.floor(pos / float(ps))
    off = pos - pidx * float(ps)
    P = table.shape[1]
    sel = (pidx.unsqueeze(-1) == torch.arange(P, device=pos.device, dtype=torch.float32)).to(torch.float32)
    page = (sel * table.to(torch.float32).unsqueeze(1)).sum(-1)  # [R, Q]
    return (page * float(ps) + off).to(torch.int64)


def _mine(x: torch.Tensor, N: int, dp_index: torch.Tensor | None) -> torch.Tensor:
    """This rank's group's rows of a group-major [N * B, ...] tensor (all of them without DP attention)."""
    if N == 1 or dp_index is None:
        return x
    B = x.shape[0] // N
    return x.reshape(N, B, *x.shape[1:]).index_select(0, dp_index).reshape(B, *x.shape[1:])


# neuronx-cc 2.27 misreads the board in a ONE-row reader graph (R = 1: no DP attention, decode bucket 1) at k = 1:
# spec_prep's draft column and mtp_prep's acc and base came back 0, so every overlapped EAGLE-3 k 1 draft was rejected
# (Llama-3.1-8B-Instruct TP 4 on trn2.3xlarge: 0 of 128 per prompt against ~0.75 synchronous; trn1.2xlarge, the graphs
# alone: R = 1 wrong at k 1, R = 2 and R = 16 right, k 3 right, board widths 7, 8 and 16 alike; SDK 2.32, 2026-10-07).
# The readers therefore run a one-row call on the row twice and return the first copy; every other shape traces as
# before.


def spec_prep(board, slot_idx, table, rows, pad_slot, valid, Q: int, k: int, ps: int, N: int, dp_index=None):
    """board [S, W] fp32; slot_idx [R] int64 (R = N * B, group-major; padding rows: a scratch slot); table [B, P]
    int64 and rows [B, Q] int64 (this rank's group); pad_slot [B, Q] int64 (KV slots for padding rows);
    valid [R] fp32 (1 for a real row). Returns ids [R, Q] int64, draft [R * Q] int64, and this group's
    positions [B, Q], KV slots [B, Q] and state rows [B, 1 + Q] (int64)."""
    if slot_idx.shape[0] == 1:  # see above: the row twice, the first copy back
        ids, draft, pos, slot, st = spec_prep(board, slot_idx.repeat(2), table.repeat(2, 1), rows.repeat(2, 1),
                                              pad_slot.repeat(2, 1), valid.repeat(2), Q, k, ps, N, dp_index)
        return ids[:1], draft[:Q], pos[:1], slot[:1], st[:1]
    sb = board.index_select(0, slot_idx)  # [R, W]
    T, acc, base, cur, drafts = sb[:, :Q], sb[:, Q], sb[:, Q + 1], sb[:, Q + 2], sb[:, Q + 4:Q + 4 + k]
    row0 = (T * _onehot(acc, Q)).sum(-1, keepdim=True)  # the newest token: T[acc]
    ids = torch.cat([row0, drafts], dim=1)  # [R, Q]
    pos = base.unsqueeze(1) + torch.arange(Q, device=board.device, dtype=torch.float32).unsqueeze(0)
    draft = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], dim=1)  # row j checks the token at j + 1
    pos_g, cur_g, valid_g = _mine(pos, N, dp_index), _mine(cur, N, dp_index), _mine(valid, N, dp_index)
    pos_g = torch.where(valid_g.unsqueeze(1) > 0, pos_g, torch.zeros_like(pos_g))  # padded rows at 0, as the host pads
    slot_g = torch.where(valid_g.unsqueeze(1) > 0, _slots(pos_g, table, ps), pad_slot)
    state_g = torch.cat([torch.where(valid_g > 0, cur_g, rows[:, 0].to(torch.float32)).unsqueeze(1).to(torch.int64),
                         rows], dim=1)
    return (ids.to(torch.int64), draft.reshape(-1).to(torch.int64), pos_g.to(torch.int64), slot_g,
            state_g)


def spec_post(board, slot_idx, out, ids, rows_all, valid, Q: int, k: int):
    """After the verify: out [R * Q, VERIFY_COLS] (engine/sampler.verify_sample: y, accept, r, ...), ids [R, Q]
    the verify's inputs, rows_all [R, Q] fp32 every row's state rows. Writes T, acc, base, cur of each real
    row's slot (padding rows write their scratch slot); returns [R, Q + 2] fp32 (T, acc, nd: whether the row had
    drafts) for the host."""
    R = ids.shape[0]
    o = out.reshape(R, Q, -1)
    y, accept, rep = o[:, :, 0], o[:, :, 1], o[:, :, 2]
    sb = board.index_select(0, slot_idx)
    real = sb[:, Q + 3] > 0  # the drafts are an MTP pass's (else a draftless row: the sample y_0)
    a = (accept[:, :k] > 0).to(torch.float32)
    run, acc = torch.ones_like(a[:, 0]), torch.zeros_like(a[:, 0])
    for j in range(k):  # the accepted prefix (an unrolled cumprod: k is static)
        run = run * a[:, j]
        acc = acc + run
    acc = acc * real.to(torch.float32)  # [R]
    oh = _onehot(acc, Q)
    new = torch.where(real, torch.where(acc < k, (rep * oh).sum(-1), y[:, k]), y[:, 0])  # replacement or bonus
    ar = torch.arange(Q, device=out.device, dtype=torch.float32).unsqueeze(0)
    nxt = torch.cat([ids[:, 1:].to(torch.float32), torch.zeros_like(y[:, :1])], dim=1)  # the drafts, shifted
    T = torch.where(ar < acc.unsqueeze(1), nxt,
                    torch.where(ar == acc.unsqueeze(1), new.unsqueeze(1), torch.zeros_like(nxt)))
    base = sb[:, Q + 1] + 1.0 + acc
    cur = (rows_all * oh).sum(-1)  # the state after the last accepted position: the request's row acc
    keep = (valid > 0).unsqueeze(1)
    row = torch.where(keep, torch.cat([T, acc.unsqueeze(1), base.unsqueeze(1), cur.unsqueeze(1), sb[:, Q + 3:]], 1), sb)
    board.index_put_((slot_idx,), row)
    return torch.cat([T, acc.unsqueeze(1), real.to(torch.float32).unsqueeze(1)], dim=1)


def mtp_prep(board, slot_idx, table, pad_slot, valid, Q: int, ps: int, N: int, dp_index=None, k: int = 1):
    """After spec_post: the MTP pass over the verified positions, ids T [R, Q] int64 (global), last_index acc [R]
    int64 (global), and this group's positions base_old .. [B, Q] and KV slots [B, Q]. With k > 1 also the later
    single-position passes' positions and KV slots [B, k - 1] (DecoderForCausalLM.forward_mtp_k pos_rest /
    slot_rest): base, base + 1, ..., after the newest token's position base, as ModelRunner.mtp_launch's
    last + 1 + s."""
    if slot_idx.shape[0] == 1:  # see spec_prep: the row twice, the first copy back
        out = mtp_prep(board, slot_idx.repeat(2), table.repeat(2, 1), pad_slot.repeat(2, 1), valid.repeat(2), Q, ps,
                       N, dp_index, k)
        return tuple(o[:1] for o in out)
    sb = board.index_select(0, slot_idx)
    T, acc, base = sb[:, :Q], sb[:, Q], sb[:, Q + 1]
    first = base - 1.0 - acc
    pos = first.unsqueeze(1) + torch.arange(Q, device=board.device, dtype=torch.float32).unsqueeze(0)
    pos_g, valid_g = _mine(pos, N, dp_index), _mine(valid, N, dp_index)
    keep = valid_g.unsqueeze(1) > 0
    pos_g = torch.where(keep, pos_g, torch.zeros_like(pos_g))  # a padded row sits at position 0
    slot_g = torch.where(keep, _slots(pos_g, table, ps), pad_slot)
    out = (T.to(torch.int64), acc.to(torch.int64), pos_g.to(torch.int64), slot_g)
    if k == 1:
        return out
    rest = base.unsqueeze(1) + torch.arange(k - 1, device=board.device, dtype=torch.float32).unsqueeze(0)
    rest_g = _mine(rest, N, dp_index)
    rest_g = torch.where(keep, rest_g, torch.zeros_like(rest_g))
    rslot_g = torch.where(keep, _slots(rest_g, table, ps), pad_slot[:, : k - 1])
    return out + (rest_g.to(torch.int64), rslot_g)


def mtp_post(board, slot_idx, drafts, valid, Q: int, k: int):
    """The MTP pass's drafts [R, k] fp32 into the board's draft columns of each real row's slot."""
    sb = board.index_select(0, slot_idx)
    keep = (valid > 0).unsqueeze(1)
    row = torch.where(keep, torch.cat([sb[:, :Q + 3], torch.ones_like(sb[:, :1]), drafts.to(torch.float32)], 1), sb)
    board.index_put_((slot_idx,), row)
    return drafts


def spec_init(board, token_board, slot_idx, base, cur, valid, Q: int, k: int):
    """After a final prefill chunk: T[0] = the token the prefill sampled into token_board, acc 0, base (the
    position of that token) and cur [R] fp32 from the host, no drafts yet (the MTP pass after the chunk writes
    them, mtp_post)."""
    sb = board.index_select(0, slot_idx)
    tok = token_board.index_select(0, slot_idx)
    R = slot_idx.shape[0]
    T = torch.cat([tok.unsqueeze(1), torch.zeros(R, Q - 1, device=board.device, dtype=torch.float32)], 1)
    row = torch.cat([T, torch.zeros_like(base).unsqueeze(1), base.unsqueeze(1), cur.unsqueeze(1),
                     torch.zeros(R, 1 + k, device=board.device, dtype=torch.float32)], 1)
    board.index_put_((slot_idx,), torch.where((valid > 0).unsqueeze(1), row, sb))
    return tok


def spec_host_init(board, slot_idx, tok, base, cur, valid, Q: int, k: int):
    """A request's board row from the host before its first blind verify: its newest token tok [R] fp32 (a prompt
    token), at position base, state row cur, no drafts (that verify is draftless)."""
    sb = board.index_select(0, slot_idx)
    R = slot_idx.shape[0]
    T = torch.cat([tok.unsqueeze(1), torch.zeros(R, Q - 1, device=board.device, dtype=torch.float32)], 1)
    row = torch.cat([T, torch.zeros_like(base).unsqueeze(1), base.unsqueeze(1), cur.unsqueeze(1),
                     torch.zeros(R, 1 + k, device=board.device, dtype=torch.float32)], 1)
    board.index_put_((slot_idx,), torch.where((valid > 0).unsqueeze(1), row, sb))
    return tok


def mtp_prefill_ids(token_board, slot_idx, ids, last):
    """The MTP ids of a prefill chunk [R, C] int64 with the token after the chunk's last position (sampled by
    the prefill, not yet on the host) taken from token_board at column last [R] int64."""
    tok = token_board.index_select(0, slot_idx).to(torch.int64)
    C = ids.shape[1]
    at = (torch.arange(C, device=ids.device).unsqueeze(0) == last.unsqueeze(1))
    return torch.where(at, tok.unsqueeze(1), ids)
