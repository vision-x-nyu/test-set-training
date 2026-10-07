#!/usr/bin/env python3
"""GPT-4o blind (text-only) baseline: top-1 accuracy on each benchmark's multiple-choice questions.

Two modes:

  --rescore (default)  Score the shipped GPT-4o responses in reproduce/data/gpt4o/ (no API calls).
                       Video-MME's gold answers are not shipped, so scoring video_mme downloads its
                       annotations (about 0.4 MB) at the pinned revision. ``--check`` compares with
                       reproduce/expected_gpt4o.json.
  --run                Query the API again and write new response caches (needs ``pip install openai``
                       and OPENAI_API_KEY in the environment).

Prompts are TsT's blind prompts (TsT.evaluators.llm.data.conversion.get_blind_qa) with letter-labeled
options, preceded by an instruction to answer with the option letter only; temperature 0, at most 10
output tokens. The first whitespace-delimited token of the response is parsed with the MMMU answer
parser. A response that names no option (mostly refusals: "I'm sorry, I can't see images") is
counted as incorrect, as in the paper (--unparseable wrong, the default). --unparseable random gives
such responses a random option instead, as the MMMU evaluation code does, drawn from
random.Random(f"{seed}:{benchmark}:{id}") so each question's draw is fixed.

Examples:
  python scripts/blind_baselines.py --check
  python scripts/blind_baselines.py --unparseable random
  OPENAI_API_KEY=... python scripts/blind_baselines.py --run --benchmarks cvb --output_dir outputs/gpt4o
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO / "reproduce" / "data" / "gpt4o"
DEFAULT_EXPECTED = REPO / "reproduce" / "expected_gpt4o.json"
BENCHMARKS = ("vsi", "cvb", "mmmu", "video_mme")

MODEL_ID = "gpt-4o-2024-08-06"
SYSTEM_PROMPT = "You are a helpful assistant that answers questions accurately and concisely."
ANSWER_PREFIX = (
    "Answer immediately with ONLY the single letter (A, B, C, D, etc.) of the correct option. Do not explain.\n\n"
)
MAX_TOKENS = 10
MAX_RETRIES = 6


# ----------------------------------------------------------------------------- scoring


class _NoDraw:
    """Stands in for the fallback RNG to detect responses that name no option."""

    @staticmethod
    def choice(seq):
        return None


def score_records(records: List[dict], benchmark: str, unparseable: str = "wrong", seed: int = 0) -> Dict:
    """Top-1 accuracy (%) of cached responses; ``unparseable``: "wrong" or "random" (seeded per question)."""
    # imported here so --help works without the package installed
    from TsT.evaluators.llm.scoring import gold_letter, parse_multi_choice_response

    n_correct = n_unparseable = 0
    for r in records:
        # only the option letters matter: the parsed response is a single token
        options = [chr(65 + i) for i in range(r["n_options"])]
        gold = gold_letter(r["ground_truth"], options)
        pred = parse_multi_choice_response(r["prediction"] or "", options, _NoDraw())
        if pred is None:
            n_unparseable += 1
            if unparseable == "wrong":
                continue
            pred = random.Random(f"{seed}:{benchmark}:{r['id']}").choice([chr(65 + i) for i in range(len(options))])
        n_correct += pred == gold
    n = len(records)
    return {
        "benchmark": benchmark,
        "n": n,
        "top1_acc": 100.0 * n_correct / n,
        "n_unparseable": n_unparseable,
        "n_api_errors": sum(1 for r in records if r.get("api_error")),
    }


def add_gold_letters(records: List[dict], benchmark: str) -> List[dict]:
    """Fill in ``ground_truth`` from the benchmark itself (downloads its annotations at the pinned revision).

    The shipped Video-MME responses carry no gold answers: its terms do not allow redistributing
    any part of the dataset, so they are read from Hugging Face instead.
    """
    if all("ground_truth" in r for r in records):
        return records
    from TsT.core.benchmark import BenchmarkRegistry

    bench = BenchmarkRegistry.get_benchmark(benchmark)
    df = bench.load_data(revision=bench.resolve_revision(None, mode="llm"))
    gold = dict(zip(df[bench.id_col].astype(str), df["gt_idx"].map(lambda i: chr(65 + int(i)))))
    return [{**r, "ground_truth": gold[str(r["id"])]} if "ground_truth" not in r else r for r in records]


def read_cache(path: Path) -> List[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cache_path(cache_dir: Path, benchmark: str) -> Path:
    for name in (f"{benchmark}_mc.jsonl.gz", f"{benchmark}_mc.jsonl"):
        if (cache_dir / name).exists():
            return cache_dir / name
    raise FileNotFoundError(f"no cache for {benchmark} in {cache_dir} ({benchmark}_mc.jsonl[.gz])")


def rescore(args) -> int:
    results = []
    for bm in args.benchmarks:
        try:
            records = add_gold_letters(read_cache(cache_path(Path(args.cache_dir), bm)), bm)
        except Exception as e:  # e.g. offline: Video-MME's gold answers must be downloaded once
            print(
                f"error: could not load the {bm} gold answers ({type(e).__name__}). Video-MME's are downloaded "
                "from Hugging Face; run once with network access, or pass --benchmarks vsi cvb mmmu.",
                file=sys.stderr,
            )
            return 1
        res = score_records(records, bm, args.unparseable, args.seed)
        results.append(res)
        print(f"{bm:10s} n={res['n']:5d}  top-1 {res['top1_acc']:6.2f}  (unparseable: {res['n_unparseable']})")
    key = "unparseable_wrong" if args.unparseable == "wrong" else f"unparseable_random_seed{args.seed}"
    out = {"model": MODEL_ID, "unparseable": args.unparseable, "seed": args.seed, "results": results}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    if args.check:
        table = json.loads(Path(args.check).read_text()).get(key)
        if table is None:
            print(f"no expected values for {key} in {args.check}", file=sys.stderr)
            return 1
        bad = [
            f"{r['benchmark']}: got {r['top1_acc']:.4f}, expected {table[r['benchmark']]:.4f}"
            for r in results
            if r["benchmark"] in table and abs(r["top1_acc"] - table[r["benchmark"]]) > args.tol
        ]
        missing = [b for b in args.benchmarks if b not in table]
        if bad or missing:
            print("MISMATCH: " + "; ".join(bad + [f"{b}: no expected value" for b in missing]), file=sys.stderr)
            return 1
        print(f"OK: {len(results)} values match {Path(args.check).name} [{key}] (tol {args.tol}).")
    return 0


# ----------------------------------------------------------------------------- API run


def blind_instances(benchmark: str) -> List[dict]:
    """The multiple-choice questions of ``benchmark`` as blind prompts (pinned revision for TsT-LLM)."""
    from TsT.core.benchmark import BenchmarkRegistry
    from TsT.core.cross_validation import UnifiedCrossValidator
    from TsT.evaluators.llm.data.conversion import get_blind_qa

    bench = BenchmarkRegistry.get_benchmark(benchmark)
    df = bench.load_data(revision=bench.resolve_revision(None, mode="llm"))
    out = []
    for model in bench.get_qa_models():
        if model.format != "mc":
            continue
        target = UnifiedCrossValidator.resolve_target_column(model, None)
        for _, row in model.select_rows(df).iterrows():
            instruction, response, _answer, options = get_blind_qa(row.to_dict(), target, "mc")
            qid = row[bench.id_col]
            out.append(
                {
                    "id": qid.item() if hasattr(qid, "item") else qid,
                    "instruction": instruction,
                    "ground_truth": response,
                    "options": [str(o) for o in options],
                }
            )
    return out


def _first_token(raw: str) -> str:
    raw = (raw or "").strip()
    return raw.split()[0].rstrip(".,!?;:()") if raw else ""


async def _query_all(instances: List[dict], max_concurrent: int) -> List[str]:
    import openai
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
    sem = asyncio.Semaphore(max_concurrent)

    async def call(text):
        resp = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""

    async def one(inst):
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    return (await call(ANSWER_PREFIX + inst["instruction"])).strip()
                except retryable as e:  # transient: retry with backoff, then record (scored as unparseable)
                    if attempt == MAX_RETRIES - 1:
                        return f"ERROR: {type(e).__name__}"
                    await asyncio.sleep(2 ** (attempt + 1))
                # other errors (authentication, permissions, unknown model, bad request) stop the run

    return await asyncio.gather(*(one(inst) for inst in instances))


def run_api(args) -> int:
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print(f"error: set OPENAI_API_KEY to query {MODEL_ID} (and pip install openai)", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for bm in args.benchmarks:
        instances = blind_instances(bm)
        raws = asyncio.run(_query_all(instances, args.max_concurrent))
        records = [
            {
                "id": i["id"],
                "ground_truth": i["ground_truth"],
                "n_options": len(i["options"]),
                "prediction": _first_token(r),
                "api_error": r.startswith("ERROR"),
                "raw_output": r,
            }
            for i, r in zip(instances, raws)
        ]
        with gzip.open(out_dir / f"{bm}_mc.jsonl.gz", "wt", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        res = score_records(records, bm, args.unparseable, args.seed)
        print(
            f"{bm:10s} n={res['n']:5d}  top-1 {res['top1_acc']:6.2f}  (unparseable {res['n_unparseable']}, "
            f"failed requests {res['n_api_errors']}) -> {out_dir}"
        )
        failed += res["n_api_errors"]
    if failed:
        print(f"error: {failed} requests failed after retries; they are scored as unparseable", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--rescore", action="store_true", help="Score cached responses (default)")
    mode.add_argument("--run", action="store_true", help="Query the API and write new caches to --output_dir")
    ap.add_argument("--cache_dir", default=str(DEFAULT_CACHE), help="Response caches to rescore")
    ap.add_argument("--output_dir", default="outputs/blind_baselines", help="(--run) where to write caches")
    ap.add_argument("--max_concurrent", type=int, default=5, help="(--run) concurrent API requests")
    ap.add_argument(
        "--unparseable",
        choices=["wrong", "random"],
        default="wrong",
        help="score of a response that names no option: wrong (default; the paper's rule) or a random option",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed of --unparseable random")
    ap.add_argument("--json", help="(--rescore) write the scores here")
    ap.add_argument(
        "--check",
        nargs="?",
        const=str(DEFAULT_EXPECTED),
        help="(--rescore) compare with expected values; exit 1 on mismatch",
    )
    ap.add_argument("--tol", type=float, default=0.005, help="absolute tolerance for --check, percentage points")
    args = ap.parse_args(argv)
    if args.run:
        if args.check or args.json:
            ap.error("--check and --json apply to --rescore; rescore new caches with --cache_dir")
        return run_api(args)
    return rescore(args)


if __name__ == "__main__":
    sys.exit(main())
