# Getting started

This page covers installing the package, running TsT-RF from the command line and from Python,
and the settings that affect reproducibility. TsT-LLM uses the same command line with
`--mode llm`; [tst-llm.md](tst-llm.md) covers what differs. IBP is in [ibp.md](ibp.md). For what
the numbers mean, see [interpreting-results.md](interpreting-results.md).

## Requirements

- Python 3.10, 3.11 or 3.12. TsT-RF, IBP with TsT-RF and the reproduction scripts run on a CPU,
  with no API keys. TsT-LLM needs Linux, one NVIDIA GPU and the `llm` extra
  ([tst-llm.md](tst-llm.md#requirements)).
- Network access to the Hugging Face Hub on the first run of each benchmark.
- Disk: under 1 MB for VSI-Bench and about 0.4 GB for CV-Bench in the Hugging Face cache.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/vision-x-nyu/test-set-training.git
cd test-set-training
uv sync --frozen
```

`uv sync --frozen` installs the exact versions in `uv.lock` into `.venv/`, including the `TsT`
package itself; add `--extra llm` for TsT-LLM. Prefix commands with `uv run`, or activate the environment with
`source .venv/bin/activate`.

With pip, `constraints.txt` pins the same versions:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt .
```

The published numbers were computed with numpy 1.26.4, pandas 2.3.0 and scikit-learn 1.6.1. Other
versions run, but they can move the headline score by about a point and per-type scores
by several points (for example, scikit-learn 1.8 and later change CV-Bench's 56.14 to 56.75), so
the published values need `uv.lock` or `constraints.txt`. `python -m TsT` prints a warning when
the installed versions differ, and `summary.json` lists them under `environment.differs_from_lock`.

Check the install:

```bash
uv run python -m TsT --help     # lists the options; no download, no torch
uv run pytest                   # offline unit and release tests (about 2 minutes)
uv run pytest -m network        # golden values; downloads about 0.75 GB (all five benchmarks' annotations)
```

## Run TsT-RF from the command line

```bash
uv run python -m TsT --benchmark vsi --group_col scene_name --output_dir outputs/vsi
uv run python -m TsT --benchmark cvb --group_col image_id --output_dir outputs/cvb
```

Each run loads the benchmark's test split at its pinned revision, builds one random-forest model
per question type, runs k-fold cross-validation, and prints a table of held-out scores with the
weighted mean (every question counts once) and the macro mean (every question type counts once).
A VSI-Bench run takes about 15–20 seconds on a multi-core CPU.

| Option | Default | Meaning |
|---|---|---|
| `--benchmark`, `-b` | required | `vsi` (VSI-Bench) or `cvb` (CV-Bench) for TsT-RF; TsT-LLM also accepts `video_mme`, `mmmu` and `mmstar` |
| `--mode`, `-m` | `rf` | `rf` (TsT-RF) or `llm` (TsT-LLM; see [tst-llm.md](tst-llm.md)) |
| `--feature_set` | `default` | `default` is leak-free and recommended. `paper` (VSI-Bench only) reproduces the proceedings' feature list, which reads held-out answers; it prints a warning on every run. |
| `--group_col` | none | Keep all questions that share this column's value in one fold: `scene_name` for VSI-Bench, `image_id` for CV-Bench. Without it, folds are drawn question by question. |
| `--revision` | pinned | Hugging Face dataset revision (commit, tag or branch). Defaults: VSI-Bench `bc96b17` (`d7cb1a3` with `--mode llm`, as in the paper's TsT-LLM runs), CV-Bench `bc284db`. |
| `--n_splits`, `-k` | 5 | Number of folds |
| `--repeats`, `-r` | 1 | Repeat the whole cross-validation with seeds `random_state + i` |
| `--random_state`, `-s` | 42 | Seed for the fold split and the forests |
| `--fold_seed` | `--random_state` | Seed for the fold split only, to measure how much a score depends on which questions share a fold |
| `--question_types`, `-q` | all | Comma-separated model names, as in the "question type" column of the results table (VSI-Bench: `object_counting`, `object_abs_distance`, `object_size_estimation`, `room_size_estimation`, `object_rel_distance`, `object_rel_direction`, `route_planning`, `obj_appearance_order`; CV-Bench: `count_2d`, `relation_2d`, `depth_3d`, `distance_3d`). The dataset's `object_rel_direction_easy`, `_medium` and `_hard` select the one `object_rel_direction` model, which scores all three. Any unknown name is an error. |
| `--target_col`, `-t` | per type | Override the column each model predicts |
| `--output_dir`, `-o` | none | Write `summary.json` and `s_x.csv` there |
| `--keep_going` | off | If one question type fails, record it and continue; the exit status is still 1 |
| `--verbose`, `-v` | off | Log per-fold scores and feature importances |

Exit status: 0 on success; 1 if any question type failed or if the loaded data do not match the
requested dataset revision; 2 on invalid arguments (including unknown question types). A failure
never produces a silently averaged score: without `--keep_going` the run stops, and with it the
failed types are listed and excluded from the means.

## Outputs

`--output_dir DIR` writes two files:

- `DIR/summary.json`: the run settings (package version, dataset repo and revision, number of rows,
  the id column, a SHA-1 fingerprint of the row order, feature set, fold settings, seed), the
  weighted and macro means, and for each question type its score, standard deviation over folds,
  fold scores and feature list, plus the Python and library versions. Scores are fractions in
  [0, 1]. `weighted_std` is the spread of the per-type scores around the weighted mean (how much
  question types differ), not an error bar; `macro_se` and `macro_t_95ci` (which reflect only the
  fold split and forest seed) are the standard error
  and 95% t-interval of the macro mean over repeats (empty with one repeat).
- `DIR/s_x.csv`: one row per question with its held-out score s(x). Columns: the benchmark's id
  column (`id` for VSI-Bench, `idx` for CV-Bench, named as in the Hugging Face dataset; recorded as
  `id_col` in `summary.json`), `model`, `question_type`, `format`, `metric`, `n_options`,
  `chance`, `s`, `s_minus_chance`, `n_repeats`. TsT-LLM runs add `s_zero_shot` and `delta`.

[interpreting-results.md](interpreting-results.md) explains both files.

`scripts/rf_report.py` runs the same evaluation and writes a compact per-question-type JSON report,
for example `uv run python scripts/rf_report.py --benchmark vsi --group_col scene_name --out vsi.json`.

## Use it from Python

```python
from TsT import evaluate_benchmark

run = evaluate_benchmark("vsi", group_col="scene_name", show_progress=False)
print(f"{100 * run.weighted_mean:.2f}")     # 38.82
sx = run.s_x()                              # the s_x.csv table as a DataFrame
run.save("outputs/vsi")                     # summary.json + s_x.csv
```

`evaluate_benchmark` accepts the same settings as the command line (`feature_set`, `revision`,
`n_splits`, `repeats`, `random_state`, `question_types`, `keep_going`) and a pre-loaded `df`. For
your own data, see [your-benchmark.md](your-benchmark.md).

## Reproducibility

- **Pinned revisions.** Shuffled k-fold assignment depends on the order of rows. VSI-Bench changed
  its row order on 2025-11-11 (same questions); at the pinned `bc96b17` the paper preset gives
  43.50, and at later revisions such as `d7cb1a3` it gives 43.64. Every load is checked against
  the revision's known row-order fingerprint (`row_order_sha1`; [reproducing.md](reproducing.md) lists them), and a
  mismatch stops the run with an error. A run at a revision other than the pinned one prints the
  fingerprint it loaded. `summary.json` records the revision and `row_order_sha1` so two runs can
  be compared.
- **Seeds.** Folds and forests are seeded from `--random_state`. With the locked versions and the
  pinned revisions, the commands in [reproducing.md](reproducing.md) reproduce the published values to two decimals.
- **Repeats.** `--repeats 5` averages over five fold splits and reports a 95% t-interval for the
  macro mean over repeats.
  The paper's TsT-RF numbers use one repeat with seed 42.

## Caching and downloads

- Benchmark files are cached per dataset revision under `HF_HOME` (default `~/.cache/huggingface`).
  Once a revision is cached, runs at that revision also work without network access (for example
  with `HF_HUB_OFFLINE=1`): the loader reads only that revision's cached files, never another
  cached version of the dataset. If a load does not match the revision's row-order fingerprint, the
  run stops; delete the dataset from the cache and run again with network access.
- CV-Bench's test files embed the images, so its first download is about 405 MB; TsT-RF uses only
  the text and metadata columns.
- If a download stalls or fails, retry with `HF_HUB_DISABLE_XET=1`.
