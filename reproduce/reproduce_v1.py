#!/usr/bin/env python3
"""Zero-GPU reproduction of the VSI-Bench-Debiased v1 evaluation numbers.

Re-aggregates cached per-sample lmms-eval predictions on VSI-Bench against the
released v1 removal list. No GPU, no model download, no API calls.

Reproduces (table numbers are the same in the proceedings and the arXiv version):
  * Table 2  fine-tuned LLaVA-Video-7B row (vision / blind / gap, original and v1)
  * Table 2/3 base LLaVA-Video-7B *blind* cells; the base *vision* cells come from a
    run on a pre-release version of VSI-Bench and are not shipped
  * Table 9  per-type composition of v1 (from the removal list)
  * Table 11 matched random-pruning control at 54% removal, 10 seeds
  * Table 12 all seven rows (rows 3-7 are 2026 lmms-eval runs, including a
    separate run of the base LLaVA-Video-7B)

Scoring: MC types use the logged `accuracy`. NUM answers are re-scored from each
`prediction` and `ground_truth` with `robust_num.py` (first number, number words and
units handled; mean relative accuracy over thresholds 0.50-0.95), for every model. The
shipped records also carry the logging fork's `MRA:.5:.95:.05` field, computed with a bare
float(prediction): it scores answers such as "0.6." or "1.5 meters." as 0, and it differs
from robust_num only for the base LLaVA-Video-7B and LLaVA-OneVision-7B blind runs.
Overall scores are MICRO averages (mean over questions).
Samples are joined on the HF `id` field, never on row position.

Usage:
  python reproduce/reproduce_v1.py                  # print tables
  python reproduce/reproduce_v1.py --check          # compare with expected_v1.json; exit 1 on mismatch
  python reproduce/reproduce_v1.py --json out.json
Requires: Python >= 3.9, numpy (tested 1.26.4) and text2digits (tested 0.1.0; used by robust_num.py).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREDS = os.path.join(HERE, "data", "preds")
DEFAULT_REMOVED = os.path.join(HERE, "data", "vsi_bench_debiased_v1_removed_ids.txt")
DEFAULT_EXPECTED = os.path.join(HERE, "expected_v1.json")

NUM_TYPES = {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}
MC_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
SEEDS = list(range(10))

# (key, vision file, blind file); files may be .jsonl or .jsonl.gz
MODELS = [
    ("llava_video_7b_ft_vsi_train_10k", "vsi_train_10k", "vsi_train_10k_blind"),
    ("cambrian_s_prerelease", "cambrian-s", "cambrian-s_blind"),
]

# Table 12 rows 3-7: (key, vision file, blind file, printed name)
TRANSFER = [
    ("internvl3_9b", "internvl3_9b", "internvl3_9b_blind", "InternVL3-9B"),
    ("internvl2_5_26b", "internvl2_5_26b", "internvl2_5_26b_blind", "InternVL2.5-26B"),
    ("internvl2_5_8b", "internvl2_5_8b", "internvl2_5_8b_blind", "InternVL2.5-8B"),
    ("llava_onevision_7b", "llava_ov_7b", "llava_ov_7b_blind", "LLaVA-OneVision-7B"),
    ("llava_video_7b_base_2026", "llava_vid_7b_2026", "llava_vid_7b_2026_blind", "LLaVA-Video-7B (2026 run)"),
]
NUM_FIELD = "robust"  # every NUM answer is re-scored with robust_num.py


def _open(path):
    for p in (path + ".jsonl", path + ".jsonl.gz"):
        if os.path.exists(p):
            return gzip.open(p, "rt") if p.endswith(".gz") else open(p)
    raise FileNotFoundError(path + ".jsonl[.gz]")


def load_scores(path, num_field="MRA:.5:.95:.05"):
    """id(str) -> (question_type, score) for scoreable samples.

    num_field="robust" scores NUM questions from `prediction` and `ground_truth` with
    robust_num.robust_mra (needs text2digits) instead of reading a logged score field.
    """
    rescore = None
    if num_field == "robust":
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from robust_num import robust_mra as rescore
    out = {}
    with _open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            doc = rec.get("doc", rec)  # raw lmms-eval log or slim record
            qt = doc.get("question_type")
            if qt in NUM_TYPES:
                v = rescore(doc.get("prediction"), doc.get("ground_truth"), qt) if rescore else doc.get(num_field)
            elif qt in MC_TYPES:
                v = doc.get("accuracy")
            else:
                continue
            if v is None:
                continue
            out[str(doc["id"])] = (qt, float(v))
    return out


def micro(scores, ids):
    vals = [scores[i][1] for i in ids if i in scores]
    return 100.0 * float(np.mean(vals)), len(vals)


def vision_blind(vis, bld, removed):
    """Vision, blind and gap on the original set and on v1 (questions scored in both runs)."""
    full = sorted(set(vis) & set(bld))
    kept = [i for i in full if i not in removed]
    r = {
        "n_full": len(full),
        "n_v1": len(kept),
        "vision_full": micro(vis, full)[0],
        "blind_full": micro(bld, full)[0],
        "vision_v1": micro(vis, kept)[0],
        "blind_v1": micro(bld, kept)[0],
    }
    r["gap_full"] = r["vision_full"] - r["blind_full"]
    r["gap_v1"] = r["vision_v1"] - r["blind_v1"]
    return r


def random_control(vis, bld, removed, seeds=SEEDS):
    """Gap on the guided subset vs. random subsets with the same removal count."""
    common = sorted(set(vis) & set(bld))  # string sort; fixes the RNG-to-sample mapping
    v = np.array([vis[i][1] for i in common])
    b = np.array([bld[i][1] for i in common])
    n = len(common)
    drop = [k for k, i in enumerate(common) if i in removed]
    keep = np.ones(n, bool)
    keep[drop] = False
    gaps = []
    for s in seeds:
        rm = np.random.default_rng(s).choice(n, size=len(drop), replace=False)
        m = np.ones(n, bool)
        m[rm] = False
        gaps.append(100.0 * (v[m].mean() - b[m].mean()))
    return {
        "n": n,
        "n_removed": len(drop),
        "full_gap": 100.0 * (v.mean() - b.mean()),
        "guided_gap": 100.0 * (v[keep].mean() - b[keep].mean()),
        "random_mean": float(np.mean(gaps)),
        "random_std_ddof1": float(np.std(gaps, ddof=1)),
        "random_min": float(np.min(gaps)),
        "random_max": float(np.max(gaps)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", default=DEFAULT_PREDS)
    ap.add_argument("--removed-ids", default=DEFAULT_REMOVED)
    ap.add_argument("--json", help="write results JSON here")
    ap.add_argument(
        "--check",
        nargs="?",
        const=DEFAULT_EXPECTED,
        metavar="EXPECTED_JSON",
        help="compare against an expected-results JSON (default: expected_v1.json); exit 1 on mismatch",
    )
    ap.add_argument("--tol", type=float, default=0.005, help="absolute tolerance, percentage points")
    a = ap.parse_args(argv)

    with open(a.removed_ids) as f:
        removed = {ln.strip() for ln in f if ln.strip()}
    res = {"removed_ids": len(removed)}

    for key, vf, bf in MODELS:
        vis = load_scores(os.path.join(a.preds_dir, vf), num_field=NUM_FIELD)
        bld = load_scores(os.path.join(a.preds_dir, bf), num_field=NUM_FIELD)
        r = vision_blind(vis, bld, removed)
        r["random_control_54pct"] = random_control(vis, bld, removed)
        res[key] = r

    # Table 12 rows 3-7.
    for key, vf, bf, _ in TRANSFER:
        vis = load_scores(os.path.join(a.preds_dir, vf), num_field=NUM_FIELD)
        bld = load_scores(os.path.join(a.preds_dir, bf), num_field=NUM_FIELD)
        res[key] = vision_blind(vis, bld, removed)

    # Base LLaVA-Video-7B, 2025 run (Tables 2/3): blind only; its vision run used a pre-release
    # VSI-Bench and is not shipped. The 2026 run of the same model (Table 12) gives the same blind scores.
    bld = load_scores(os.path.join(a.preds_dir, "llava_vid_7b_blind"), num_field=NUM_FIELD)
    ids = sorted(bld)
    res["llava_video_7b_base_blind"] = {
        "blind_full": micro(bld, ids)[0],
        "blind_v1": micro(bld, [i for i in ids if i not in removed])[0],
    }

    # Table 9: v1 composition by question type.
    qt_of = {i: qt for i, (qt, _) in load_scores(os.path.join(a.preds_dir, "vsi_train_10k")).items()}
    orig, rem = Counter(qt_of.values()), Counter(qt_of[i] for i in removed if i in qt_of)
    res["v1_composition"] = {
        qt: {"original": orig[qt], "removed": rem[qt], "kept": orig[qt] - rem[qt]}
        for qt in sorted(orig, key=lambda q: -orig[q])
    }
    res["v1_composition_total"] = {"original": sum(orig.values()), "removed": sum(rem.values())}

    _print(res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
    if a.check:
        with open(a.check) as f:
            exp = json.load(f)
        bad = _compare(exp, res, a.tol)
        if bad:
            print("\nMISMATCH:")
            for b in bad:
                print("  " + b)
            return 1
        print(f"\nOK: all {_count(exp)} expected values match (tol {a.tol}).")
    return 0


def _flatten(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, pre + k + ".")
        else:
            yield pre + k, v


def _count(d):
    return sum(1 for _ in _flatten(d))


def _compare(exp, got, tol):
    flat = dict(_flatten(got))
    bad = []
    for k, v in _flatten(exp):
        g = flat.get(k)
        if g is None:
            bad.append(f"{k}: missing")
        elif isinstance(v, float) and not abs(g - v) <= tol:  # NaN fails
            bad.append(f"{k}: expected {v:.4f}, got {g:.4f}")
        elif not isinstance(v, float) and g != v:
            bad.append(f"{k}: expected {v}, got {g}")
    return bad


def _print(res):
    ft, cs = res["llava_video_7b_ft_vsi_train_10k"], res["cambrian_s_prerelease"]
    print(f"removed ids: {res['removed_ids']}")
    print("\nTable 2 / 12 (micro, %)        Vis    Blind  Gap   | v1 Vis  Blind  Gap   | dGap")
    rows = [("Cambrian-S (pre-release ckpt)", cs), ("LLaVA-Video-7B + VSI-Train-10k", ft)]
    rows += [(name, res[key]) for key, _, _, name in TRANSFER]
    for name, r in rows:
        print(
            f"{name:30s} {r['vision_full']:5.2f}  {r['blind_full']:5.2f}  {r['gap_full']:5.2f} | "
            f"{r['vision_v1']:5.2f}  {r['blind_v1']:5.2f}  {r['gap_v1']:5.2f} | {r['gap_v1'] - r['gap_full']:+.2f}"
        )
    b = res["llava_video_7b_base_blind"]
    print(
        f"Base LLaVA-Video-7B blind, 2025 run (Tables 2/3): {b['blind_full']:.2f} -> {b['blind_v1']:.2f} "
        "(vision: pre-release VSI-Bench run, not shipped)"
    )
    print("\nTable 11 (54% removal, 10 seeds)  full   guided  random mean+/-sd [min,max]")
    for name, r in (("LLaVA-Video-7B + FT", ft), ("Cambrian-S", cs)):
        c = r["random_control_54pct"]
        print(
            f"{name:32s} {c['full_gap']:5.2f}  {c['guided_gap']:5.2f}   {c['random_mean']:5.2f} +/- "
            f"{c['random_std_ddof1']:.2f} [{c['random_min']:.2f},{c['random_max']:.2f}]  (n={c['n']}, rm={c['n_removed']})"
        )
    print("\nTable 9 composition: type original removed kept")
    for qt, c in res["v1_composition"].items():
        print(f"  {qt:28s} {c['original']:5d} {c['removed']:5d} {c['kept']:5d}")
    t = res["v1_composition_total"]
    print(f"  {'TOTAL':28s} {t['original']:5d} {t['removed']:5d} {t['original'] - t['removed']:5d}")


if __name__ == "__main__":
    sys.exit(main())
