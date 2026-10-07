# Running TsT on your own benchmark

TsT-RF needs three things from you: the test set as a table, one feature-based model per question
type, and a check that no feature reads a held-out question's answer.
[`examples/custom_benchmark.py`](../examples/custom_benchmark.py) does all three on a toy benchmark
in a few seconds:

```bash
uv run python examples/custom_benchmark.py                   # ends with "label invariance: PASS"
uv run python examples/custom_benchmark.py --group_col scene_id
uv run python examples/custom_benchmark.py --demo-leak       # plants a leaky feature; the check fails
```

In the toy data, `taller_object` answers follow real-world object heights 85% of the time and
`object_count` answers follow per-object ranges, so both are learnable without the image;
`object_color` answers are uniformly random, so TsT-RF stays near chance there.

## 1. Load the test set

One row per question, with at least:

- a unique question id (`id`), used to join s(x) back to your data;
- `question_type`: one random-forest model is trained per type, because features are type-specific;
- the question text, and the answer options for multiple-choice (MC) questions;
- the gold answer, in the column the models predict (the target);
- `n_options` for MC questions, so the per-question export can report chance;
- optionally a grouping column (scene, video, image or source document) for grouped folds.

Keep the row order fixed (for example, pin your dataset revision). Shuffled k-fold assignment
depends on it.

## 2. Write one feature-based model per question type

A model implements [`FeatureBasedBiasModel`](../src/TsT/core/protocols.py):

```python
from TsT.core.protocols import FeatureBasedBiasModel

class MyTypeModel(FeatureBasedBiasModel):
    name = "my_type"          # matches question_type
    format = "mc"             # "mc" -> classifier, accuracy; "num" -> regressor, MRA
    feature_cols = ["opt_0_freq", "opt_1_freq"]

    def select_rows(self, df):            # this type's rows; keep df's index
        ...
    def fit_feature_maps(self, train_df): # statistics from the TRAINING fold only
        ...
    def add_features(self, df):           # a copy of df with feature_cols added
        ...
```

For each fold, TsT calls `fit_feature_maps` on the training rows, then `add_features` on the
training and held-out rows, trains a forest on the training features, and scores the held-out rows.
Object-valued feature columns are label-encoded on the training fold.

Good features describe what a text-only model could know: entities parsed from the question,
option properties (count, position, length, numeric value), and statistics of the training fold,
such as how often each option text was the answer or the typical value of each object's counts.

**The one rule:** `add_features` must not read the row's own answer. Statistics learned in
`fit_feature_maps` from training answers are fine; looking up a held-out row's statistics with its
own gold answer is not. That bug is easy to write (for example, "the frequency of the gold option's
object", where the gold option comes from the row's answer) and it inflates the score: the toy
example's `--demo-leak` raises `taller_object` from about 0.68 to about 0.88.

Run the models with `run_evaluation`:

```python
from TsT.evaluation import per_question_scores, run_evaluation

models = [MyTypeModel(), OtherTypeModel()]
results = run_evaluation(question_models=models, df_full=df, n_splits=5, random_state=42,
                         target_col="answer", group_col="scene_id")
sx = per_question_scores(df, models, results, id_col="id")   # the s_x.csv table
```

`run_evaluation` raises if a model fails (pass `keep_going=True` to record the failure and continue).
Each result carries the held-out score, the fold scores, per-question held-out predictions and the
forest's feature importances.

## 3. Check label invariance

Fit the feature maps on a training split, compute features for the held-out rows, change only the
held-out answers (MC: another valid letter; NUM: another value), and compute the features again.
Any feature column that changes is keyed on the held-out label. `check_label_invariance` in the
example implements this in about 15 lines; copy it into your tests and run it for every model and
every feature preset you ship.

The check catches features that read the answer directly. It cannot tell you whether a
legitimately computed feature reflects a real-world prior or a construction artifact of your
benchmark; that judgement is the point of reviewing high-s(x) questions
([interpreting-results.md](interpreting-results.md)).

## 4. Choose the folds

Question-level folds (the default) let a model learn from other questions about the same scene or
image. If your questions share a source, also run grouped folds (`group_col`) and report both: a
large drop means part of the signal is shared context within a source rather than a property of
the question template.

## 5. Optional: register a benchmark

To get `summary.json`, `s_x.csv` and the command-line interface for your benchmark, subclass
[`Benchmark`](../src/TsT/core/benchmark.py) the way [`benchmarks/vsi`](../src/TsT/benchmarks/vsi)
and [`benchmarks/cvb`](../src/TsT/benchmarks/cvb) do: set `name`, `hf_repo`, `default_revision`
and `id_col`, implement `load_data(revision)` and `get_feature_based_models(feature_set)`, and
decorate the class with `@BenchmarkRegistry.register`. For a Hugging Face dataset, `load_data` can
call `TsT.utils.load_hf_split(hf_repo, revision)` and then `self.check_row_order(df, revision)`,
with the expected fingerprints in `row_order_fingerprints`, as the built-in benchmarks do. Then
`evaluate_benchmark(MyBenchmark(), df=df).save("outputs/mine")` writes both files. To expose it as
`python -m TsT --benchmark <name>`, add the module under `src/TsT/benchmarks/` and its name to
`BUILTIN_BENCHMARKS` in `src/TsT/core/benchmark.py`.

## 6. TsT-LLM on your benchmark

TsT-LLM needs no features, only prompts. Give each row a `question` (text), an `options` list
for MC questions, a `question_format` (`mc` or `num`) and the gold answer: an option letter in
`ground_truth`, or an option index in `gt_idx`. Then add `get_qa_models` to your benchmark class:

```python
from TsT.core.qa_models import MCBenchmarkQAModel, NumericalBenchmarkQAModel

    def get_qa_models(self):
        return [
            MCBenchmarkQAModel(benchmark_name=self.name),          # trained to answer the option letter
            NumericalBenchmarkQAModel(benchmark_name=self.name),   # trained to answer the number
        ]
```

The models read the answer from `ground_truth` by default; `MCBenchmarkQAModel(...,
default_target_col="gt_idx")` reads it from `gt_idx` instead. Prompts are truncated at
`LLMRunConfig.max_seq_length` tokens (1024 by default), so very long questions need a larger value.
Options are labelled "A.", "B.", ... unless an option's text already starts with its own letter.
Run it with `evaluate_benchmark(MyBenchmark(), mode="llm", df=df)`, inside an
`if __name__ == "__main__":` block when it is in a script (vLLM may spawn a process that
re-imports the script), on a GPU machine with the
`llm` extra ([tst-llm.md](tst-llm.md)). The prompts are in
[`evaluators/llm/data/conversion.py`](../src/TsT/evaluators/llm/data/conversion.py).
