"""Device-resident recurrent state of linear-attention layers: rows for running requests, and
state checkpoints for the prefix cache.

A linear-attention layer (models/linear_attn.py) keeps, per sequence, the last K - 1 inputs
of its short conv and one [Dk, Dv] fp32 matrix per head, instead of KV pages. Each such layer
gets two pools on the device, conv [rows, K - 1, conv_dim] (model dtype) and recurrent
[rows, heads, Dk, Dv] (fp32, as transformers' mamba_ssm_dtype float32), and every graph call
passes a state_slot per sequence: decode reads and writes its row once per token, every
prefill chunk carries the row forward, so chunked prefill and decode compose. Row 0 is the
scratch row that padded batch rows read and write.

Rows are not reset on the host. A sequence whose first computed position is 0 starts from
zero state inside the graph (linear_attn.mixer), so a request admitted into a reused row, or
recomputed from scratch after a preemption, never sees its predecessor's state.

Speculative decoding: a request holds `rows_per_req` = 1 + k rows. The verify graph reads its
CURRENT row and writes the state after each of its 1 + k positions into its rows in order; the
row after the last accepted position becomes current (vLLM v0.30.0
v1/attention/backends/gdn_attn.py: num_spec + 1 state slots per request, the next step reads
slot num_accepted_tokens - 1), so a rejected draft costs no copy.

State checkpoints (the prefix cache of these models): the rows past the running ones hold the
state of all linear layers after a page-aligned prefix, one per radix node that has one
(engine/radix_cache.py RadixNode.ckpt; SGLang v0.5.21 mem_cache/unified_cache/components/
mamba.py keeps one state per node the same way). They are written and read by copying whole
rows on the device (ModelRunner.copy_state).

DP attention (engine/dp.py): every rank's pools hold the rows of its own group's requests and
checkpoints only, so rank 0 hands out rows from free lists per group (a request's group is
req.dp_group, a checkpoint's the group whose radix cache owns it); the same row number names a
different request or checkpoint on each group's ranks.
"""

from __future__ import annotations

import torch

from .request import Status

SCRATCH_ROW = 0


class StatePool:
    def __init__(self, model, rows: int, device: torch.device, conv_dtype: torch.dtype, groups: int = 1,
                 rows_per_req: int = 1, ckpt_rows: int = 0):
        """rows: requests that can run at once (per DP-attention group), each holding rows_per_req
        rows; ckpt_rows rows more (per group) for state checkpoints; and the scratch row."""
        self.per_req = rows_per_req
        n_run = rows * rows_per_req
        self.rows = 1 + n_run + ckpt_rows
        shapes = model.state_shapes()
        self.conv = [torch.zeros((self.rows, *c), dtype=conv_dtype, device=device) for c, _ in shapes]
        self.rec = [torch.zeros((self.rows, *r), dtype=torch.float32, device=device) for _, r in shapes]
        model.bind_state(self.conv, self.rec)
        # Other per-request rows a model asks for (a Per-Layer Embedding's conv history,
        # models/hybrid.py aux_state_shapes), managed with the same rows.
        self.aux = [torch.zeros((self.rows, *a), dtype=d, device=device) for a, d in model.aux_state_shapes()]
        model.bind_aux_state(self.aux)
        # Running rows go out in sets of per_req consecutive ids, from one free list per DP group.
        self._frees = [[list(range(1 + i * rows_per_req, 1 + (i + 1) * rows_per_req)) for i in range(rows - 1, -1, -1)]
                       for _ in range(groups)]
        self._held: dict[str, list] = {}  # rid -> [rows, current index, request]
        self.num_ckpt_rows = ckpt_rows
        self._ckpt_frees = [list(range(self.rows - 1, n_run, -1)) for _ in range(groups)]

    def pools(self) -> list[torch.Tensor]:
        return self.conv + self.rec + self.aux

    def rows_of(self, req) -> list[int]:
        """The request's rows, allocated on first use. When none are free, rows held by requests
        of its group that are not running (finished, or preempted and waiting) are reclaimed:
        every call is built from running requests only, so a running request's rows are never
        taken, and a reclaimed holder restarts from zero state or a checkpoint if it ever runs
        again. A finished holder's last step may still be in flight; it executes before any later
        call."""
        held = self._held.get(req.rid)
        if held is not None:
            return held[0]
        g = req.dp_group
        free = self._frees[g]
        if not free:
            for rid, (rs, _, other) in list(self._held.items()):
                if other.status is not Status.RUNNING and other.dp_group == g:
                    del self._held[rid]
                    free.append(rs)
        if not free:
            raise RuntimeError(f"recurrent-state pool exhausted ({len(self._held)} requests hold rows)")
        rs = free.pop()
        self._held[req.rid] = [rs, 0, req]
        return rs

    def row(self, req) -> int:
        """The request's current row: the one its next call reads and (outside verify) writes."""
        rs = self.rows_of(req)
        return rs[self._held[req.rid][1]]

    def set_current(self, req, i: int) -> None:
        """After a verify: row i of the request's rows holds the state its next call starts from."""
        self.rows_of(req)
        self._held[req.rid][1] = i

    def release(self, req) -> None:
        held = self._held.pop(req.rid, None)
        if held is not None:
            self._frees[held[2].dp_group].append(held[0])

    def alloc_ckpt(self, group: int = 0) -> int | None:
        free = self._ckpt_frees[group]
        return free.pop() if free else None

    def free_ckpt(self, row: int, group: int = 0) -> None:
        self._ckpt_frees[group].append(row)

    def free_ckpt_rows(self, group: int = 0) -> list[int]:
        return self._ckpt_frees[group]

    @property
    def num_free_ckpts(self) -> int:
        return len(self._ckpt_frees[0])

    def bytes_per_row(self) -> int:
        return sum(t[0].numel() * t.element_size() for t in self.pools())
