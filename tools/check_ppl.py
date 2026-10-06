"""Mean prompt logprob of natural sentences, on the device as served: a working model scores
ordinary English around -1 to -3 nats per token, a broken one far lower. Cheap enough to
compare checkpoint interpretations (env knobs KILN_QKV_LAYOUT, KILN_MXFP4_NIBBLES).

    python tools/check_ppl.py --model XiaomiMiMo/MiMo-V2.6-Flash-RL --tp 32 --piecewise
"""

from __future__ import annotations

import argparse

import torch

TEXTS = [
    "The capital of France is Paris, and the capital of Italy is Rome.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "def add(a, b):\n    return a + b\n",
    "The quick brown fox jumps over the lazy dog.",
]

# --long: one passage of ordinary English (written for this check, about 1000 tokens) scored through
# several prefill chunks, so a linear-attention model's chunked state carry (and the NKI delta-rule
# kernel's 128-token chunks, KILN_LINEAR_ATTN_KERNEL=nki) is exercised on real weights.
LONG_TEXT = " ".join([
    "Water moves through the environment in a continuous cycle. Heat from the sun warms the surface of",
    "oceans, lakes and rivers, and some of that water turns into vapor and rises into the air. As the",
    "vapor climbs, the air around it cools, and the vapor condenses into tiny droplets that gather into",
    "clouds. When the droplets grow large enough, they fall back to the ground as rain or snow. Some of",
    "that water runs off the land into streams, some soaks into the soil and becomes groundwater, and",
    "some is taken up by plants, which release it again through their leaves. Over time, nearly all of",
    "it finds its way back to the sea, and the cycle begins again. The same water that fell as rain on",
    "an ancient forest may today be part of a glacier, a cup of tea, or a cloud drifting over a city.",
    "The first computers filled entire rooms. They were built from vacuum tubes, which glowed like light",
    "bulbs and failed often, and engineers spent much of their time replacing the ones that burned out.",
    "Programs were entered by setting switches or by feeding in cards with holes punched in them. The",
    "invention of the transistor changed everything: it did the same job as a tube, but it was smaller,",
    "used less power and lasted far longer. Later, many transistors were placed on a single chip of",
    "silicon, and the number that could fit on one chip doubled roughly every two years. Machines that",
    "once needed a building of their own shrank to the size of a desk, then a book, and finally a phone",
    "that fits in a pocket and is more powerful than the room-sized computers that came before it.",
    "A good loaf of bread needs only four ingredients: flour, water, salt and yeast. The baker mixes them",
    "into a dough and kneads it, stretching and folding it until it becomes smooth and elastic. Kneading",
    "develops gluten, a network of proteins that traps the gas the yeast produces as it feeds on the",
    "sugars in the flour. The dough is then left to rise in a warm place, often for several hours, and",
    "during that time it slowly doubles in size. After it is shaped and has risen once more, it goes",
    "into a hot oven. The heat makes the gas expand one last time, sets the structure of the crumb, and",
    "browns the outside into a crisp crust. Many bakers say that patience matters more than any recipe.",
    "Trains have connected distant towns for almost two hundred years. The early locomotives burned coal",
    "to boil water, and the steam pushed pistons that turned the wheels. Railways allowed farmers to send",
    "their crops to markets far away and let people travel in a day a distance that once took a week.",
    "Today many trains run on electricity drawn from wires above the track, and the fastest ones carry",
    "passengers at more than three hundred kilometers per hour. Even so, the basic idea has not changed:",
    "steel wheels rolling on steel rails waste very little energy, which is why a single locomotive can",
    "pull a long line of heavy cars across a continent while using far less fuel than a fleet of trucks.",
    "Bees are small, but they play an enormous role in feeding the world. As a bee visits flowers to",
    "collect nectar and pollen, grains of pollen stick to the fine hairs on its body and are carried to",
    "the next flower, which allows the plant to produce seeds and fruit. Apples, almonds, berries and",
    "many vegetables depend on this help. A single colony can contain tens of thousands of workers, and",
    "each one has a job that changes as it grows older: cleaning the hive, feeding the young, building",
    "wax comb, guarding the entrance, and finally flying out to forage. Together they behave almost like",
    "one large animal, sharing food and information and keeping the temperature of the hive steady.",
])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--core-base", type=int, default=0, help="first NeuronCore (tp_core_base)")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--piecewise", action="store_true")
    ap.add_argument("--piecewise-group", type=int, default=6)
    ap.add_argument("--kv-cache-gb", type=float, default=1.0)
    ap.add_argument("--dp-attention", type=int, default=1, help="DP-attention groups (EngineConfig.dp_attention)")
    ap.add_argument("--no-vocab-parallel", action="store_true", help="replicate embedding and lm_head")
    ap.add_argument("--reference-json", default=None, help="tools/hf_reference.py output to compare with")
    ap.add_argument("--weight-dtype", default="auto", choices=["auto", "bf16", "fp8", "fp8-experts"])
    ap.add_argument("--mxfp4-packed", action="store_true")
    ap.add_argument("--long", action="store_true", help="score LONG_TEXT through 256-token prefill chunks instead")
    ap.add_argument("--text-file", default=None,
                    help="score this file's text instead (its first --max-tokens tokens, 256-token prefill chunks)")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--chunk", type=int, default=None,
                    help="--text-file: tokens per prefill call (default 256), so --chunk 1024 at --dp-attention 4 runs the "
                         "sequence-parallel row gathers at 32 rows per rank (the served graphs' path at 128)")
    # The target with an MTP head loaded (its weights, KV, and the MTP graph after each prefill chunk): the
    # prompt logprobs are the target's, so they must not move. --state-checkpoints holds the recurrent-state
    # pool at the baseline's rows (bench/serve_sweep.py --state-checkpoints): 1 + 4 x (1 + k) + checkpoints.
    ap.add_argument("--spec-method", default=None, choices=["mtp"])
    ap.add_argument("--spec-k", type=int, default=1)
    ap.add_argument("--state-checkpoints", type=int, default=None)
    ap.add_argument("--out-json", default=None,
                    help="write the token ids, each position's logprob and top-1 id (tools/compare_ppl.py reads it)")
    return ap


def _long_shape(args) -> tuple[int, int]:
    """(prefill chunk, max_model_len) of a --long / --text-file run, (32, 512) otherwise."""
    if getattr(args, "text_file", None):
        return getattr(args, "chunk", None) or 256, -(-(args.max_tokens + 64) // 256) * 256
    return (256, 2048) if getattr(args, "long", False) else (32, 512)


def engine_config(args, path: str | None = None):
    """The EngineConfig this check runs with (tools/compile_farm.py capture --tool check_ppl builds
    the same one; it runs prompt logprobs, so capture with --plp)."""
    from kiln.config import EngineConfig
    from kiln.models.loader import resolve_model_path

    chunk, max_len = _long_shape(args)
    return EngineConfig(
        model_path=path or resolve_model_path(args.model), device=args.device, dtype=torch.bfloat16, page_size=32,
        max_num_seqs=4, max_model_len=max_len, max_prefill_tokens=chunk, kv_cache_gb=args.kv_cache_gb,
        decode_batch_buckets=(-(-4 // args.dp_attention),), prefill_token_buckets=(chunk // args.dp_attention,),
        page_buckets=(4, max_len // 32), tp=args.tp,
        tp_core_base=args.core_base, dp_attention=args.dp_attention,
        piecewise=args.piecewise, vocab_parallel=not args.no_vocab_parallel, piecewise_group=args.piecewise_group,
        weight_dtype=args.weight_dtype, mxfp4_packed=args.mxfp4_packed,
        **({"spec_method": args.spec_method, "spec_k": args.spec_k} if getattr(args, "spec_method", None) else {}),
        **({"state_checkpoints": args.state_checkpoints} if getattr(args, "state_checkpoints", None) is not None else {}))


def main() -> None:
    args = build_parser().parse_args()
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    texts = [LONG_TEXT] if args.long else TEXTS
    if args.text_file:
        with open(args.text_file) as f:
            texts = [f.read()]
    long_run = args.long or args.text_file
    chunk = _long_shape(args)[0]
    eng = LLMEngine(engine_config(args))
    ids = [eng.tokenizer(t)["input_ids"] for t in texts]
    if args.text_file:
        ids = [ids[0][: args.max_tokens]]
    reqs = eng.generate(ids, SamplingParams(max_new_tokens=1, prompt_logprobs=1 if args.out_json else 0))
    eng.close()
    if args.out_json:
        import json

        with open(args.out_json, "w") as f:
            json.dump([{"ids": i, "logprobs": [r.prompt_logprobs[q][0] for q in range(1, len(i))],
                        "top1": [r.prompt_logprobs[q][2][0] for q in range(1, len(i))]} for i, r in zip(ids, reqs)], f)
    ref = None
    if args.reference_json:
        import json

        with open(args.reference_json) as f:
            ref = json.load(f)
    total = n = 0
    for i, (t, r) in enumerate(zip(texts, reqs)):
        lps = [v[0] for v in r.prompt_logprobs.values()]
        total, n = total + sum(lps), n + len(lps)
        want = f"  reference {ref['ppl'][i]['mean_logprob']:7.3f}" if ref else ""
        print(f"  {sum(lps) / len(lps):7.3f}{want}  {t[:50]!r}" + (f" ({len(lps) + 1} tokens)" if long_run else ""))
    extra = f" reference={ref['mean_prompt_logprob']:.3f}" if ref else ""
    print(f"RESULT mean_prompt_logprob={total / n:.3f}{extra} tokens={n}")
    if long_run:  # by prefill chunk: the state carried across chunk boundaries
        lps = [v[0] for _, v in sorted(reqs[0].prompt_logprobs.items())]
        print("by chunk: " + " ".join(f"{sum(lps[i:i + chunk]) / len(lps[i:i + chunk]):.3f}"
                                      for i in range(0, len(lps), chunk)))


if __name__ == "__main__":
    main()
