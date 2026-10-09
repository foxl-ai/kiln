"""tools/ep_lanes.py: the EP prefill kernel's executed lanes against routed pairs, on host routing of real text (s3
logs/kiln-mimo-trn1/ep_routing.pt: GLM-5.3-Flash's routers on wikitext-2, 2 sequences x 4096 tokens, every MoE layer).
Contiguous placement (rank r: experts 9r .. 9r + 8), 288 experts, top-8, 32 ranks, C = 4096 rows per call.
Pass plan (kernels/moe_ep.py, C >= 2048): every local expert one static pass of LW = 256 lanes, then past its first
256 pairs nb = floor(r / 512) + [r mod 512 > 256] passes of 512 lanes and ns = [left > 0] one of 256.
Per lane 1568 moving columns (gate_up 1024, down 512, x transposes 32): PE time = lanes x 1568 / 2.45 G cols/s.

    python tools/ep_lanes.py ep_routing.pt        # R8 calls: one text's 4096 tokens; G64: 4 x 1024-token chunks
"""
import sys, torch
d = torch.load(sys.argv[1])
topi, names, E = d["topi"], d["names"], d["experts"]
layers = sorted(topi)
texts = [i for i, n in enumerate(names) if not n.startswith("random")]
print("sequences", names, "text", [names[i] for i in texts], "layers", len(layers), "experts", E)
R, El, LW, LW2 = 32, E // 32, 256, 512
def lanes(n):
    r = max(n - LW, 0); nb = r // LW2 + (1 if r % LW2 > LW else 0); left = r - nb * LW2
    return LW + nb * LW2 + (LW if left > 0 else 0)
def call_stats(rows_topi):
    """rows_topi [4096, k] -> per rank (pairs, lanes, static-pass padding, overflow padding)"""
    cnt = torch.bincount(rows_topi.long().flatten(), minlength=E)
    out = []
    for r in range(R):
        ne = cnt[r * El:(r + 1) * El].tolist()
        sp = sum(LW - min(n, LW) for n in ne)
        la = sum(lanes(n) for n in ne)
        out.append((sum(ne), la, sp, la - sum(ne) - sp))
    return out
for shape in ("R8", "G64"):
    calls = []
    for s in texts:
        T = topi[layers[0]][s].shape[0]
        if shape == "R8":
            calls.append([(s, 0, T)])
    if shape == "G64":
        a, b = texts[0], texts[1]
        calls = [[(a, 0, 1024), (a, 1024, 2048), (b, 0, 1024), (b, 1024, 2048)],
                 [(a, 2048, 3072), (a, 3072, 4096), (b, 2048, 3072), (b, 3072, 4096)]]
    tot = {"r0_pairs": 0, "r0_lanes": 0, "max_pairs": 0, "max_lanes": 0, "mean_pairs": 0, "mean_lanes": 0, "n": 0,
           "max_sp": 0, "max_op": 0, "mean_sp": 0, "mean_op": 0}
    for call in calls:
        for l in layers:
            rows = torch.cat([topi[l][s][a:b] for s, a, b in call])
            st = call_stats(rows)
            tot["r0_pairs"] += st[0][0]; tot["r0_lanes"] += st[0][1]
            mx = max(st, key=lambda x: x[1])
            tot["max_pairs"] += mx[0]; tot["max_lanes"] += mx[1]; tot["max_sp"] += mx[2]; tot["max_op"] += mx[3]
            tot["mean_sp"] += sum(x[2] for x in st) / R; tot["mean_op"] += sum(x[3] for x in st) / R
            tot["mean_pairs"] += sum(x[0] for x in st) / R; tot["mean_lanes"] += sum(x[1] for x in st) / R
            tot["n"] += 1
    n = tot["n"]
    for who in ("r0", "mean", "max"):
        p, la = tot[f"{who}_pairs"] / n, tot[f"{who}_lanes"] / n
        print(f"{shape} {who:4s} rank per MoE layer: pairs {p:7.1f}, executed lanes {la:7.1f}, padded share {1 - p / la:.1%}, "
              f"PE floor {la * 1568 / 2.45e9 * 1e3:.3f} ms executed / {p * 1568 / 2.45e9 * 1e3:.3f} ms routed"
              + (f"; padding in static passes {tot[who + '_sp'] / n:.0f} lanes, in overflow rounding {tot[who + '_op'] / n:.0f}"
                 if who != "r0" else ""))
