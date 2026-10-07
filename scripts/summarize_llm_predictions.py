#!/usr/bin/env python3
"""Summarize TsT-LLM per-question predictions (``<output_dir>/predictions/`` of a ``--mode llm`` run).

For each question-answer model (e.g. vsi_mc, vsi_num) it reports, for the zero-shot pass,
each fold, and all folds pooled:
  - score: the score TsT reports (mean gold-option probability for MC, MRA for NUM). The
    pooled fold score equals the run's overall score.
  - top1_acc (MC only): accuracy of the most probable option letter. When no option letter
    is among the top log-probabilities, the generated first token is used if it is an
    option letter; otherwise the question counts as wrong.
  - n_gold_not_in_topk (MC only): questions whose gold letter was not among the top
    log-probabilities, so their score came from parsing the generated text.
and the TsT - zero-shot differences (delta_score, delta_top1_acc).

Usage: python scripts/summarize_llm_predictions.py outputs/vsi_llm/predictions [--out summary.json]
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


def _letters(options):
    return [chr(65 + i) for i in range(len(options or []))]


def _top1(rec):
    probs = rec.get("option_probs")
    if probs:
        return max(probs.items(), key=lambda kv: kv[1])[0]
    pred = (rec.get("prediction") or "").strip().rstrip(".,!?;:)").lstrip("(")
    return pred if pred in _letters(rec.get("options")) else None


def _summ(recs):
    out = {"n": len(recs), "score": sum(r["score"] for r in recs) / len(recs) if recs else float("nan")}
    if recs and recs[0].get("options"):
        out["top1_acc"] = sum(1 for r in recs if _top1(r) == r["ground_truth"].strip()) / len(recs)
        out["n_gold_not_in_topk"] = sum(1 for r in recs if r.get("confidence") is None)
    return out


def summarize(pred_dir):
    by_model = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        name = os.path.basename(path)[: -len(".jsonl")]
        m = re.match(r"(.+)_(zero_shot|seed\d+_fold\d+)$", name)
        if not m:
            continue
        with open(path) as f:
            by_model[m.group(1)][m.group(2)] = [json.loads(line) for line in f if line.strip()]

    summary = {}
    for model, parts in by_model.items():
        folds = sorted(k for k in parts if k != "zero_shot")
        res = {"zero_shot": _summ(parts["zero_shot"])} if "zero_shot" in parts else {}
        res["folds"] = {k: _summ(parts[k]) for k in folds}
        pooled = [r for k in folds for r in parts[k]]
        if pooled:
            res["tst_pooled"] = _summ(pooled)
            if "zero_shot" in res:
                res["delta_score"] = res["tst_pooled"]["score"] - res["zero_shot"]["score"]
                if "top1_acc" in res["tst_pooled"]:
                    res["delta_top1_acc"] = res["tst_pooled"]["top1_acc"] - res["zero_shot"]["top1_acc"]
        summary[model] = res
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    summary = summarize(args.pred_dir)
    if not summary:
        print(
            f"error: no prediction files (<model>_zero_shot.jsonl, <model>_seed<S>_fold<K>.jsonl) in {args.pred_dir}; "
            "run python -m TsT --mode llm with --output_dir",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
