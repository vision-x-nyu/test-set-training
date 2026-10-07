# Iterative Bias Pruning (IBP)

IBP turns TsT's per-question scores into a pruned benchmark (Sec. 4, Algorithm 1). Each iteration
runs TsT on the remaining questions, ranks them by their held-out score, and removes the top batch,
until the removal budget is spent or the highest remaining score falls to an early-stop threshold.
Re-running TsT after every batch lets the ranking adapt as questions are removed.

The pruned set depends on the diagnostic, the allocation and the seed (Appendix D.4 and D.5). Treat a
removal list as an audit selection to inspect, not as the single correct debiased benchmark.

## Run it

TsT-RF (CPU; VSI-Bench and CV-Bench):

```bash
python -m TsT.debiasing --benchmark vsi --alloc per_format --budget 1000 --output_dir outputs/ibp_vsi_pf_b1000
```

TsT-LLM (one GPU, the `llm` extra; see [tst-llm.md](tst-llm.md)). The paper's MMMU analysis:

```bash
python -m TsT.debiasing --benchmark mmmu --mode llm --alloc global --budget 100 --batch_size 20 \
    --early_stop 0.05 --output_dir outputs/ibp_mmmu
```

Python:

```python
from TsT.debiasing import debias_benchmark

run = debias_benchmark("vsi", mode="rf", alloc="per_format", budget=1000)
run.save("outputs/ibp_vsi_pf_b1000")
print(run.n_removed, len(run.kept_ids))
```

Cost: each iteration is a full TsT run on the remaining questions, and there are about
budget / batch_size iterations per group. TsT-RF iterations take seconds; with TsT-LLM each one is a
zero-shot pass plus k LoRA trainings, so the MMMU command above takes about two hours on one GPU.
Use a larger `--batch_size` on large benchmarks.

Every TsT run inside IBP goes through `TsT.evaluation.evaluate_benchmark`, so the TsT options
(`--revision`, `--feature_set`, `--n_splits`, `--random_state`, `--group_col`, the `--llm_*`
settings) mean the same as for `python -m TsT`. The dataset is loaded once, at the benchmark's
pinned revision unless `--revision` is given, and checked against its row-order fingerprint.

## Outputs

`--output_dir` receives:

- `removed_ids.txt`: the removed questions' ids (the benchmark's id column), one per line, in
  removal order;
- `kept_ids.txt`: the remaining ids, in dataset order;
- `summary.json`: the settings, dataset revision and row-order fingerprint, the budget of each
  group, and for every iteration the number of questions, the maximum and mean bias score at the
  start of the iteration, and the ids removed.

Join on the ids, never on row positions: a dataset revision can reorder rows.

## Allocation

`--alloc` decides how the budget is split. Each group is pruned on its own, with TsT run on that
group's questions only.

| `--alloc` | Groups | Budget |
|---|---|---|
| `global` | all questions | `--budget` |
| `per_format` | one per answer format (`mc`, `num`) | `--budget`, split in proportion to group size (largest remainder), or explicit `--budgets mc=100,num=100` |
| `per_type` | one per question type | `--frac F` removes round(F x n) from every type, `--budgets_from FILE` uses the per-type counts of the ids in FILE, or explicit `--budgets TYPE=N,...` |

Per-format pruning ranks MC and NUM questions separately because their scores are on different
scales (P(gold) vs. MRA). Within MC it ranks raw P(gold) across question types, as in the paper, so
types with fewer options are removed first (at B=1000 on VSI-Bench, 44% of the two-option easy
relative-direction questions versus about 5% of the four-option types); use `--alloc per_type` for
type-balanced selections. `--alloc global` on a benchmark with both formats ranks P(gold) against
MRA, which mostly removes NUM questions. Within a group, the benchmark's strategy keeps a minimum number of
questions per question type (VSI-Bench and CV-Bench: 10; MMMU: 5 per subfield; `--min_per_group`
overrides it); `per_type` budgets are capped so the floor can be met.

## What IBP ranks by

- `--score s` (default for TsT-RF): the held-out score s(x), P(gold) for MC questions and the MRA of
  the prediction for NUM questions.
- `--score delta` (default for TsT-LLM): s(x) minus the base model's zero-shot score on the same
  question, the per-question gain from fine-tuning on the other folds. The paper's MMMU analysis and
  its TsT-LLM removal set rank by delta.

Questions that no model scores (MMMU's 53 open-ended questions under TsT-LLM) are never removed;
with `--alloc global` they count toward the group floors, and the per-format and per-type budgets
are split over the scored questions only. A question that a model should score but has no score is an error.

## Ties

Scores tie often: at the first iteration on VSI-Bench, 303 numerical questions score exactly 1.0,
and Random Forest probabilities take few distinct values. By default ties are broken by question
id (a stable sort on (-score, id), with scores compared at 12 decimals so that floating-point noise
in the last bits cannot reorder them), so given the scores the ranking does not depend on row order
or on the sort implementation (the scores themselves depend on row order through the shuffled folds,
so the dataset revision is pinned). `--tie_break legacy` uses pandas' default unstable sort on the
score alone, as the proceedings' runs did; it exists only to re-derive those removal sets, and it
re-derives them exactly only on x86-64 CPUs where numpy uses its AVX-512 sort (elsewhere, including
Apple Silicon, about three quarters of the per-format B=1000 set matches). Tie handling changes about a
quarter of the per-format B=1000 removal set, so the paper's v2 numbers use the id tie-break.

## Reproducing the paper's IBP results

The removal lists behind Tables 6, 10, 11 and 12 are shipped in `reproduce/data/ibp/`, and
`python reproduce/ibp_tables.py --check` re-derives the tables from them on CPU in about a second.
`uv run bash reproduce/regenerate_ibp.sh outputs/ibp 5` regenerates the TsT-RF lists with this CLI
(about 10 minutes with 5 parallel runs, 30 one at a time) and checks them against the shipped ones. See [reproduce/README.md](../reproduce/README.md).
