"""Bring your own benchmark: TsT-RF on a toy text-only benchmark, on CPU, in seconds.

`toy.jsonl` holds 360 templated questions about imaginary rooms, in three question types:

* ``taller_object`` (2-option MC): the answer follows real-world object heights 85% of the
  time, so a text-only model can learn the shortcut from the test set itself;
* ``object_count`` (numerical): each object's count is drawn from its own range, another
  learnable prior;
* ``object_color`` (4-option MC): the answer is uniformly random, so there is no shortcut
  and TsT-RF should stay near chance.

The script defines one feature-based model per question type, cross-validates a random
forest on each (k folds, every question scored by a model that never saw it), prints the
held-out scores and the highest-s(x) questions, and finally runs a label-invariance check:
features computed for held-out rows must not change when only those rows' answers change.

    uv run python examples/custom_benchmark.py                 # question-level folds
    uv run python examples/custom_benchmark.py --group_col scene_id
    uv run python examples/custom_benchmark.py --demo-leak     # plant a leaky feature; the check fails

See docs/your-benchmark.md for a walkthrough.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEFGHIJ"  # option letters (the toy data use up to four)


# =============================================================================
# 1. Load the benchmark into a DataFrame (one row per question)
# =============================================================================


def load_toy(path: Path = HERE / "toy.jsonl") -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["question_format"] = np.where(df["options"].notna(), "mc", "num")
    df["n_options"] = df["options"].apply(lambda o: len(o) if isinstance(o, list) else 0)
    return df.reset_index(drop=True)


def option_texts(options) -> list[str]:
    """'A. guitar' -> 'guitar'."""
    return [o.split(". ", 1)[-1] for o in options]


# =============================================================================
# 2. One feature-based model per question type
#
# Contract (TsT.core.protocols.FeatureBasedBiasModel):
#   select_rows(df)        -> the rows of this question type (keep the original index)
#   fit_feature_maps(train)-> learn statistics from the TRAINING fold only
#   add_features(df)       -> a copy of df with `feature_cols` added, using only the row's
#                             question/options and the statistics from fit_feature_maps.
#                             Never read the row's own answer here.
# =============================================================================

from TsT.core.protocols import FeatureBasedBiasModel  # noqa: E402


class TallerObjectModel(FeatureBasedBiasModel):
    """2-option MC. Feature: how often each option object was the answer in the training fold."""

    name = "taller_object"
    format = "mc"
    feature_cols = ["opt_0_win_rate", "opt_1_win_rate", "win_rate_diff"]

    def __init__(self, leaky: bool = False):
        self.win_rate: dict[str, float] = {}
        self.leaky = leaky
        if leaky:  # a bug of the kind TsT-RF features must avoid: it reads the row's own answer
            self.feature_cols = self.feature_cols + ["gold_option_win_rate"]

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = df[df["question_type"] == self.name].copy()
        opts = qdf["options"].apply(option_texts)
        qdf["opt_0"], qdf["opt_1"] = opts.str[0], opts.str[1]
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        seen, won = {}, {}
        for o0, o1, ans in zip(train_df["opt_0"], train_df["opt_1"], train_df["answer"]):
            winner = (o0, o1)[LETTERS.index(ans)]
            for o in (o0, o1):
                seen[o] = seen.get(o, 0) + 1
            won[winner] = won.get(winner, 0) + 1
        # Laplace-smoothed win rate; unseen objects get 0.5
        self.win_rate = {o: (won.get(o, 0) + 1) / (n + 2) for o, n in seen.items()}

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["opt_0_win_rate"] = df["opt_0"].map(self.win_rate).fillna(0.5)
        df["opt_1_win_rate"] = df["opt_1"].map(self.win_rate).fillna(0.5)
        df["win_rate_diff"] = df["opt_0_win_rate"] - df["opt_1_win_rate"]
        if self.leaky:
            gold = [r[f"opt_{LETTERS.index(a)}_win_rate"] for (_, r), a in zip(df.iterrows(), df["answer"])]
            df["gold_option_win_rate"] = gold
        return df


class ObjectCountModel(FeatureBasedBiasModel):
    """Numerical. Features: the training fold's count statistics for the asked-about object."""

    name = "object_count"
    format = "num"
    feature_cols = ["obj_mean", "obj_median", "obj_std", "obj_n", "global_mean"]

    def __init__(self):
        self.stats: pd.DataFrame | None = None
        self.global_mean = float("nan")

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = df[df["question_type"] == self.name].copy()
        qdf["object"] = qdf["question"].str.extract(r"How many (.+?)s are in")[0]
        qdf["answer"] = pd.to_numeric(qdf["answer"])  # numerical target
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        g = train_df.groupby("object")["answer"]
        self.stats = pd.DataFrame(
            {"obj_mean": g.mean(), "obj_median": g.median(), "obj_std": g.std().fillna(0.0), "obj_n": g.size()}
        )
        self.global_mean = float(train_df["answer"].mean())

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["obj_mean", "obj_median", "obj_std", "obj_n"]:
            df[col] = df["object"].map(self.stats[col])
        df[["obj_mean", "obj_median"]] = df[["obj_mean", "obj_median"]].fillna(self.global_mean)
        df[["obj_std", "obj_n"]] = df[["obj_std", "obj_n"]].fillna(0.0)
        df["global_mean"] = self.global_mean
        return df


class ObjectColorModel(FeatureBasedBiasModel):
    """4-option MC with a uniformly random answer: there is nothing to learn from text alone."""

    name = "object_color"
    format = "mc"
    feature_cols = ["object", *[f"opt_{i}_color" for i in range(4)], *[f"opt_{i}_color_rate" for i in range(4)]]

    def __init__(self):
        self.color_rate: dict[str, float] = {}

    def select_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        qdf = df[df["question_type"] == self.name].copy()
        qdf["object"] = qdf["question"].str.extract(r"What color is the (.+?)\?")[0]
        opts = qdf["options"].apply(option_texts)
        for i in range(4):
            qdf[f"opt_{i}_color"] = opts.str[i]
        return qdf

    def fit_feature_maps(self, train_df: pd.DataFrame) -> None:
        golds = [option_texts(o)[LETTERS.index(a)] for o, a in zip(train_df["options"], train_df["answer"])]
        offered = pd.Series([c for o in train_df["options"] for c in option_texts(o)]).value_counts()
        chosen = pd.Series(golds).value_counts()
        self.color_rate = (chosen.reindex(offered.index).fillna(0) / offered).to_dict()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for i in range(4):
            df[f"opt_{i}_color_rate"] = df[f"opt_{i}_color"].map(self.color_rate).fillna(0.25)
        return df


# =============================================================================
# 3. Label-invariance self-check (copy this into your own benchmark's tests)
# =============================================================================


def perturb_answers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Change ONLY the answers: MC -> a different valid letter; NUM -> another value."""
    df = df.copy()
    for i, row in df.iterrows():
        if row["question_format"] == "mc":
            k = row["n_options"]
            df.at[i, "answer"] = LETTERS[(LETTERS.index(row["answer"]) + 1 + int(rng.integers(k - 1))) % k]
        else:
            df.at[i, "answer"] = str(int(float(row["answer"])) + int(rng.integers(1, 10)))
    return df


def check_label_invariance(model, df: pd.DataFrame, trials: int = 3, seed: int = 0) -> list[str]:
    """Return the feature columns whose held-out values depend on the held-out answers."""
    rng = np.random.default_rng(seed)
    rows = df[df["question_type"] == model.name]
    leaky: set[str] = set()
    for _ in range(trials):
        test_ids = rows.index[rng.random(len(rows)) < 0.2]
        train, test = df.drop(index=test_ids), df.loc[test_ids]
        model.fit_feature_maps(model.select_rows(train))
        a = model.add_features(model.select_rows(test))
        b = model.add_features(model.select_rows(perturb_answers(test, rng)))
        for c in model.feature_cols:
            if not a[c].astype(str).equals(b[c].astype(str)):
                leaky.add(c)
    return sorted(leaky)


# =============================================================================
# 4. Run TsT-RF and report
# =============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group_col", default=None, help="keep rows sharing this column in one fold, e.g. scene_id")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo-leak", action="store_true", help="add a feature that reads the held-out answer")
    ap.add_argument("--verbose", action="store_true", help="show progress bars")
    args = ap.parse_args()

    from TsT.evaluation import per_question_scores, run_evaluation

    t0 = time.time()
    df = load_toy()
    models = [TallerObjectModel(leaky=args.demo_leak), ObjectCountModel(), ObjectColorModel()]

    folds = f"grouped by {args.group_col}" if args.group_col else "question-level"
    print(f"TsT-RF on the toy benchmark: {len(df)} questions, {args.n_splits} folds ({folds})\n")
    results = run_evaluation(  # raises if a question type fails
        question_models=models,
        df_full=df,
        n_splits=args.n_splits,
        random_state=args.seed,
        target_col="answer",
        group_col=args.group_col,
        show_progress=args.verbose,
    )

    chance = {"taller_object": "0.50", "object_count": "-", "object_color": "0.25"}
    print(f"{'question_type':<15} {'format':<6} {'metric':<6} {'n':>4} {'held-out':>9} {'chance':>7}")
    for r in results:
        print(
            f"{r.model_name:<15} {r.model_format:<6} {r.metric_name:<6} {r.count:>4} {r.overall_mean:>9.3f} "
            f"{chance[r.model_name]:>7}"
        )
    total = sum(r.count for r in results)
    weighted = sum(r.overall_mean * r.count for r in results) / total
    print(f"{'weighted':<15} {'':<6} {'':<6} {total:>4} {weighted:>9.3f}\n")

    # The same per-question table the CLI writes to s_x.csv: s(x) is P(gold) for MC and MRA for NUM.
    sx = per_question_scores(df, models, results, id_col="id").merge(df[["id", "question"]], on="id")
    top = sx[sx["question_type"] == "taller_object"].nlargest(3, "s_minus_chance")
    print("highest s(x) - chance within taller_object (most shortcut-solvable without the image):")
    for _, r in top.iterrows():
        print(f"  id {r['id']:>3}  s(x)={r['s']:.2f}  {r['question']}")
    print()

    bad = {m.name: check_label_invariance(m, df) for m in models}
    bad = {k: v for k, v in bad.items() if v}
    for name, cols in bad.items():
        print(f"  {name}: features keyed on the held-out answer: {cols}")
    print(f"label invariance: {'FAIL' if bad else 'PASS'}")
    print(f"done in {time.time() - t0:.1f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
