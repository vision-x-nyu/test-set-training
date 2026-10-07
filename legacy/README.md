# Legacy: the script that built VSI-Bench-Debiased v1

`debias_vsi_clean.py` is the script that produced VSI-Bench-Debiased v1, a designer-in-the-loop
pilot built with hand-written per-type filters and hand-set budgets. It predates the automated
Iterative Bias Pruning (IBP) procedure in the paper (Algorithm 1) and is not its output; it does not
compute or use TsT bias scores s(x). The automated IBP code is in
[`src/TsT/debiasing`](../src/TsT/debiasing) ([docs/ibp.md](../docs/ibp.md)).

- **What it does.** Hand-written filters, one per question type, score each test
  question from answer statistics (for example, how often an object–answer pair
  occurs, how close an answer sits to its category's mean, or answer imbalance).
  Hand-set per-type budgets then remove 2,768 of the 5,130 questions. In the code,
  the filters read only the test questions' text, options and ground-truth answers,
  and model scores enter only the optional summary table (`--ref_evals`).
- **What is canonical.** The released removal list,
  [`reproduce/data/vsi_bench_debiased_v1_removed_ids.txt`](../reproduce/data/vsi_bench_debiased_v1_removed_ids.txt),
  is byte-identical to `pruned_ids.txt` in `nyu-visionx/VSI-Bench` (revision
  `bdcadb3`). Evaluate on v1 with that list or with the `debiased` config on
  Hugging Face, not with a fresh run of this script.
- **How well it reproduces v1.** On VSI-Bench revision `bc96b17` with numpy 1.26.4,
  the script selects exactly the released 2,768 questions when numpy's AVX-512 code
  paths are off. When they are on, it matches 2,758 of them (table below).

## Run it

From the repository root, in the package environment (`uv sync --frozen`):

```bash
uv run python legacy/debias_vsi_clean.py --compare
```

The script downloads the VSI-Bench test split (about 160 KB of parquet; no videos)
at revision `bc96b17` and writes the selected IDs, one per line, to
`outputs/legacy_v1/removed_ids.txt`. It never writes over the released list.
`--compare` prints the overlap with the released list. It exits 0 for either outcome
in the table below and 1 for anything else.

| Environment (VSI-Bench `bc96b17`, numpy 1.26.4) | `--compare` result |
|---|---|
| numpy's AVX-512 code paths off: `NPY_DISABLE_CPU_FEATURES=AVX512F uv run python legacy/debias_vsi_clean.py --compare` | 2,768 selected, 2,768 overlap: **identical to the released list** |
| numpy's AVX-512 code paths on (default on x86-64 CPUs with AVX-512) | 2,768 selected, 2,758 overlap, 10 mismatches (8 object counting, 2 object size) |

Several filters sort scores that tie or nearly tie. numpy's AVX-512 code paths sort
tied scores in a different order (object counting) and change the last bits of some
floating-point scores (object size). Both outcomes were measured on the same x86-64
machine, switching only the environment variable above. CPUs without AVX-512 should
behave like the first row, but that has not been tested. The selection also depends
on:

- **numpy version.** numpy 2.5.3 gives 2,756 overlap (12 mismatches in the same two
  types) whether AVX-512 is on or off. Use the locked numpy 1.26.4.
- **Dataset revision.** Tie-breaking follows row order. The 2025-11-11 upload
  (`d7cb1a3`) reordered the rows, and at that revision the same filters overlap the
  released list on only 2,610 questions. `--revision` exists to show this; keep the
  default.

Other options:

- `--out PATH` writes the IDs somewhere else.
- `--ref_evals DIR` loads per-sample lmms-eval logs (`*.jsonl` or `*.jsonl.gz`, one
  file per model). Try `--ref_evals reproduce/data/preds`. The script then prints each
  model's overall score on the removed, original and kept questions. The table scores
  this run's selection with this script's own parsing, so use `reproduce/` for the
  paper's numbers.

## Changes from the pre-release script

The filter and scoring functions, their default weights and seeds, and the per-type
budgets are unchanged. Release changes are limited to input and output:

- The dataset revision is pinned to `bc96b17` (the original loaded the default branch),
  and the loaded rows are checked against that revision's row-order fingerprint. The
  original used `datasets.load_dataset`, which can silently read another cached
  revision when the Hugging Face Hub is unreachable or `HF_HUB_OFFLINE=1` is set.
- Output goes to `outputs/legacy_v1/removed_ids.txt`, sorted. The original wrote
  `data/removed_ids.txt` in set-iteration order.
- Reference-model evaluations are opt-in (`--ref_evals`). The original needed a local
  `data/ref_evals/` directory just to import.
- matplotlib and seaborn are imported inside the two plotting helpers, which the
  command line never calls. `text2digits` is optional.
- `--compare` is new.

## Notes

- The optional scoring helpers (`mean_relative_accuracy`, used only for
  `--ref_evals`) count a prediction as correct at threshold θ when the relative error
  is `<= 1 - θ`. Official VSI-Bench uses the same rule. The TsT package's scoring
  uses a strict `<`.
