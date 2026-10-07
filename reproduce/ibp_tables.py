#!/usr/bin/env python3
"""CPU reproduction of the Iterative Bias Pruning (IBP) results: Tables 8, 10, 13 and 14.

Re-scores shipped per-question predictions on shipped removal lists. No GPU, no model or dataset
download, no API calls; needs numpy only.

  Table 10  VSI-Bench refinement provenance: fine-tuned LLaVA-Video-7B vision-blind gap on the
            questions kept by automated TsT-RF IBP (per-format B=1000; per-type uniform 54%; per-type
            with the v1 pilot's per-type budgets, seeds 42 and 1) and by the v1 manual pilot, each with
            a matched random-removal control (10 seeds). Cambrian-S gaps are reported as well.
  Table 13  Removal rates by question type for TsT-LLM (ranks Delta s(x), budget 200) and TsT-RF
            (ranks s(x), B=1000), and the Jaccard overlap of their removed sets (also at a matched
            budget of 200).
  Table 14  Per-format TsT-RF IBP budget sweep: removal rate and mean s(x) of the remaining MC and NUM
            questions at the start of the last iteration (from the runs' summary.json files).
  Table 8   MMMU: LLaVA-OneVision-7B vision and blind accuracy on the original 900 questions and on
            the 800 kept by TsT-LLM IBP (Delta s(x), 100 removed), with 1,000 matched random removals.

Inputs (reproduce/data/):
  preds/*.jsonl.gz              VSI-Bench per-question predictions (see README.md)
  ibp/rf/<run>/                 TsT-RF IBP runs: removed_ids.txt and summary.json, written by
                                `python -m TsT.debiasing` (regenerate with regenerate_ibp.sh)
  ibp/llm/                      TsT-LLM IBP removal lists (VSI-Bench per format; MMMU)
  ibp/mmmu/*.jsonl.gz           LLaVA-OneVision-7B MMMU-val correctness per question (vision, blind)
  vsi_bench_debiased_v1_removed_ids.txt

Usage:
  python reproduce/ibp_tables.py                 # print the tables
  python reproduce/ibp_tables.py --check         # compare with expected_ibp.json; exit 1 on mismatch
  python reproduce/ibp_tables.py --json out.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reproduce_v1 import MODELS, _compare, _count, load_scores, micro, random_control  # noqa: E402

DATA = os.path.join(HERE, "data")
IBP = os.path.join(DATA, "ibp")
DEFAULT_EXPECTED = os.path.join(HERE, "expected_ibp.json")

# Table 10 rows: (key, label, removal list)
TABLE10 = [
    ("rf_per_format_b1000", "Automated per-format, B=1000", ("rf", "pf_b1000")),
    ("rf_per_type_uniform54", "Automated per-type, uniform", ("rf", "pt_uniform54")),
    ("rf_per_type_v1budgets_seed1", "Automated per-type, pilot budgets, seed 1", ("rf", "pt_v1budgets_s1")),
    ("rf_per_type_v1budgets_seed42", "Automated per-type, pilot budgets, seed 42", ("rf", "pt_v1budgets_s42")),
    ("v1_manual_pilot", "Manual pilot v1", None),
]
SWEEP = [200, 500, 1000, 1500, 2000, 2500]
TABLE13_TYPES = ["obj_appearance_order", "object_counting", "object_size_estimation", "object_abs_distance"]
MMMU_SEEDS = 1000


def read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def rf_run(name):
    d = os.path.join(IBP, "rf", name)
    with open(os.path.join(d, "summary.json")) as f:
        summary = json.load(f)
    return set(read_ids(os.path.join(d, "removed_ids.txt"))), summary


def removal_list(spec):
    if spec is None:
        return set(read_ids(os.path.join(DATA, "vsi_bench_debiased_v1_removed_ids.txt")))
    return rf_run(spec[1])[0]


def table10(preds):
    out = {}
    for key, _label, spec in TABLE10:
        removed = removal_list(spec)
        row = {}
        for model_key, (vis, bld) in preds.items():
            rc = random_control(vis, bld, removed)
            row[model_key] = {
                "n_removed": rc["n_removed"],
                "n_kept": rc["n"] - rc["n_removed"],
                "removed_pct": 100.0 * rc["n_removed"] / rc["n"],
                "gap": rc["guided_gap"],
                "random_mean": rc["random_mean"],
                "random_min": rc["random_min"],
                "random_max": rc["random_max"],
            }
        out[key] = row
    ft = preds["llava_video_7b_ft_vsi_train_10k"]
    full = sorted(set(ft[0]) & set(ft[1]))
    out["original"] = {"n_kept": len(full), "gap": micro(ft[0], full)[0] - micro(ft[1], full)[0]}
    return out


def table13(qtype):
    llm = set()
    for fmt in ("mc", "num"):
        llm |= set(read_ids(os.path.join(IBP, "llm", f"vsi_pf_{fmt}_removed_ids.txt")))
    rf = rf_run("pf_b1000")[0]
    rf200 = rf_run("pf_b200")[0]
    orig = Counter(qtype.values())
    rates = {}
    for qt in sorted(orig, key=lambda q: -orig[q]):
        rates[qt] = {
            "tst_llm_pct": 100.0 * sum(qtype[i] == qt for i in llm) / orig[qt],
            "tst_rf_pct": 100.0 * sum(qtype[i] == qt for i in rf) / orig[qt],
        }
    return {
        "n_removed_llm": len(llm),
        "n_removed_rf": len(rf),
        "intersection": len(llm & rf),
        "union": len(llm | rf),
        "jaccard": len(llm & rf) / len(llm | rf),
        "jaccard_matched_budget_200": len(llm & rf200) / len(llm | rf200),
        "removal_rates": rates,
    }


def table14():
    out = {}
    for b in SWEEP:
        _, s = rf_run(f"pf_b{b}")
        groups = {g["group"]: g for g in s["groups"]}
        out[f"budget_{b}"] = {
            "n_removed": s["n_removed"],
            "removed_pct": 100.0 * s["n_removed"] / s["n_rows"],
            "batch_size": s["batch_size"],
            "mc_mean_bias": groups["mc"]["mean_bias_last_iteration"],
            "num_mean_bias": groups["num"]["mean_bias_last_iteration"],
        }
    return out


def table8():
    def load(mode):
        with gzip.open(os.path.join(IBP, "mmmu", f"llava_ov_7b_mmmu_val_{mode}.jsonl.gz"), "rt") as f:
            return [json.loads(line) for line in f if line.strip()]

    vis_recs = load("vision")
    vis, bld = {r["id"]: r["correct"] for r in vis_recs}, {r["id"]: r["correct"] for r in load("blind")}
    open_ended = {r["id"] for r in vis_recs if r["question_type"] != "multiple-choice"}
    ids = sorted(set(vis) & set(bld))
    removed = set(read_ids(os.path.join(IBP, "llm", "mmmu_delta_removed_ids.txt")))
    kept = [i for i in ids if i not in removed]

    def acc(c, s):
        return 100.0 * sum(c[i] for i in s) / len(s)

    gaps = []
    for seed in range(MMMU_SEEDS):  # same sampling as the original analysis
        drop = set(random.Random(seed).sample(ids, len(ids) - len(kept)))
        keep = [i for i in ids if i not in drop]
        gaps.append(acc(vis, keep) - acc(bld, keep))
    return {
        "original": {
            "n": len(ids),
            "vision": acc(vis, ids),
            "blind": acc(bld, ids),
            "gap": acc(vis, ids) - acc(bld, ids),
        },
        "delta_s_ranked": {
            "n": len(kept),
            "vision": acc(vis, kept),
            "blind": acc(bld, kept),
            "gap": acc(vis, kept) - acc(bld, kept),
        },
        "random": {
            "n_seeds": MMMU_SEEDS,
            "gap_mean": statistics.mean(gaps),
            "gap_sd": statistics.pstdev(gaps),  # population s.d., as in the original analysis
            "gap_min": min(gaps),
            "gap_max": max(gaps),
        },
        "n_open_ended": len(open_ended),
        "n_removed_open_ended": len(removed & open_ended),
    }


def settings_check():
    """Settings of the shipped TsT-RF runs (all must use the paper's setup)."""
    out = {}
    for name in [f"pf_b{b}" for b in SWEEP] + ["pt_uniform54", "pt_v1budgets_s42", "pt_v1budgets_s1"]:
        _, s = rf_run(name)
        out[name] = {
            "revision": s["revision"][:7],
            "feature_set": s["feature_set"],
            "tie_break": s["tie_break"],
            "random_state": s["random_state"],
            "row_order_sha1": s["row_order_sha1"],
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write results JSON here")
    ap.add_argument("--check", nargs="?", const=DEFAULT_EXPECTED, metavar="EXPECTED_JSON")
    ap.add_argument("--tol", type=float, default=0.005, help="absolute tolerance (percentage points / score units)")
    a = ap.parse_args(argv)

    preds = {}
    for key, vf, bf in MODELS:
        preds[key] = (load_scores(os.path.join(DATA, "preds", vf)), load_scores(os.path.join(DATA, "preds", bf)))
    qtype = {i: qt for i, (qt, _) in preds["llava_video_7b_ft_vsi_train_10k"][0].items()}

    res = {
        "table10": table10(preds),
        "table13": table13(qtype),
        "table14": table14(),
        "table8": table8(),
        "rf_run_settings": settings_check(),
    }
    _print(res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
    if a.check:
        with open(a.check) as f:
            exp = json.load(f)
        exp = {k: v for k, v in exp.items() if not k.startswith("_")}
        bad = _compare(exp, res, a.tol)
        if bad:
            print("\nMISMATCH:")
            for b in bad:
                print("  " + b)
            return 1
        print(f"\nOK: all {_count(exp)} expected values match (tol {a.tol}).")
    return 0


def _print(res):
    t10 = res["table10"]
    ft, cs = "llava_video_7b_ft_vsi_train_10k", "cambrian_s_prerelease"
    print(
        "Table 10 (fine-tuned LLaVA-Video-7B; micro gap, %)   removed%   kept    gap | random (10 seeds) mean [min,max] | Cambrian-S gap / random"
    )
    print(f"  {'Original':48s} {0:7.1f}  {t10['original']['n_kept']:5d}  {t10['original']['gap']:5.2f}")
    for key, label, _ in TABLE10:
        r, c = t10[key][ft], t10[key][cs]
        print(
            f"  {label:48s} {r['removed_pct']:7.1f}  {r['n_kept']:5d}  {r['gap']:5.2f} | {r['random_mean']:5.2f} "
            f"[{r['random_min']:.2f},{r['random_max']:.2f}] | {c['gap']:5.2f} / {c['random_mean']:5.2f}"
        )
    t13 = res["table13"]
    print(
        f"\nTable 13: Jaccard {t13['jaccard']:.3f} ({t13['intersection']} shared of {t13['union']}); "
        f"matched budget 200: {t13['jaccard_matched_budget_200']:.3f}"
    )
    print(f"  {'question type':30s} TsT-LLM %  TsT-RF %")
    for qt, r in t13["removal_rates"].items():
        mark = "  (in Table 13)" if qt in TABLE13_TYPES else ""
        print(f"  {qt:30s} {r['tst_llm_pct']:8.1f}  {r['tst_rf_pct']:8.1f}{mark}")
    print("\nTable 14 (per-format, mean s(x) at the start of the last iteration)")
    print("      B  removed%   MC     NUM   batch")
    for b, r in res["table14"].items():
        print(
            f"  {b.split('_')[1]:>5s}  {r['removed_pct']:7.1f}  {r['mc_mean_bias']:.3f}  {r['num_mean_bias']:.3f}  {r['batch_size']}"
        )
    t8 = res["table8"]
    print("\nTable 8 (MMMU val, LLaVA-OneVision-7B; accuracy %)      n   vision  blind   gap")
    for key, label in (("original", "Original"), ("delta_s_ranked", "Delta s(x)-ranked removal")):
        r = t8[key]
        print(f"  {label:48s} {r['n']:4d}  {r['vision']:5.1f}  {r['blind']:5.1f}  {r['gap']:5.2f}")
    rr = t8["random"]
    print(
        f"  {'Random removal (1,000 seeds)':48s}  800      -      -   {rr['gap_mean']:.2f} +/- {rr['gap_sd']:.2f} "
        f"[{rr['gap_min']:.3f}, {rr['gap_max']:.3f}]"
    )


if __name__ == "__main__":
    sys.exit(main())
