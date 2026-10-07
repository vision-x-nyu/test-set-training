# Evaluating MLLMs on VSI-Bench and VSI-Bench-Debiased

This page is for model developers who want to report results on VSI-Bench-Debiased. No code in
this repository is needed to evaluate a model; the datasets and an lmms-eval task are public.

## The data

[`nyu-visionx/VSI-Bench`](https://hf.co/datasets/nyu-visionx/VSI-Bench) has three configs:

| Config | Questions | Contents |
|---|---|---|
| `full` (default) | 5,130 | Every VSI-Bench question, with a boolean `pruned` column |
| `debiased` | 2,362 | VSI-Bench-Debiased: the questions it keeps |
| `pruned` | 2,768 | The questions VSI-Bench-Debiased removes |

The repository also holds `pruned_ids.txt`, the 2,768 removed ids; this repository ships the same
list as [`reproduce/data/vsi_bench_debiased_v1_removed_ids.txt`](../reproduce/data/vsi_bench_debiased_v1_removed_ids.txt).

VSI-Bench-Debiased is a designer-in-the-loop pilot built with hand-written, question-type-specific
filters and hand-set budgets. It reduces non-visual shortcuts but does not remove them, and it was
not re-diagnosed with TsT after pruning. It predates the automated Iterative Bias Pruning (IBP) algorithm and is not its output.
It removes questions unevenly across types, so it changes the question-type mix:

| Question type | Original | Kept in VSI-Bench-Debiased |
|---|---:|---:|
| object_counting | 565 | 251 |
| object_abs_distance | 834 | 434 |
| object_size_estimation | 953 | 353 |
| room_size_estimation | 288 | 200 |
| object_rel_distance | 710 | 310 |
| object_rel_direction_easy | 217 | 212 |
| object_rel_direction_medium | 378 | 54 |
| object_rel_direction_hard | 373 | 116 |
| route_planning | 194 | 114 |
| obj_appearance_order | 618 | 318 |
| **Total** | **5,130** | **2,362** |

**Report VSI-Bench-Debiased alongside the full set, not in place of it.**

## Scoring

- Multiple-choice (MC) questions: accuracy. Numerical (NUM) questions: mean relative accuracy (MRA)
  over thresholds 0.50, 0.55, …, 0.95.
- The paper reports **micro averages**: every question counts once, on each set.
- lmms-eval's `vsibench` overall score is a **macro average** over question types, so its headline
  number differs from the paper's. Because VSI-Bench-Debiased changes the type mix, the two aggregations
  diverge more on it than on the full set. State which one you report.
- Join predictions to questions on `id`, never on row position: the `full` config's row order
  changed on 2025-11-11.

## With lmms-eval

[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) has the tasks
[`vsibench` and `vsibench_debiased`](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/vsibench):

```bash
python -m lmms_eval --model <model> --model_args <args> \
    --tasks vsibench,vsibench_debiased --log_samples --output_path <dir>
```

At the time of writing, lmms-eval's `vsibench_pruned` task loads the `full` config rather than the
removed questions; if you need the removed subset, use the `pruned` config or filter `full` by
`pruned`.

`--log_samples` keeps per-question records, from which you can compute micro averages.
[`reproduce/reproduce_v1.py`](../reproduce/reproduce_v1.py) is a short reference implementation: it
reads lmms-eval sample logs (or the slimmed records in `reproduce/data/`), computes micro averages,
the vision-blind gap and a matched random-removal control.

## Blind scores and the vision-blind gap

The paper's primary signal for a debiased set is the **drop in blind score**: a model's score on
the same questions without any visual input. Removing non-visual shortcuts should lower blind
scores much more than vision scores. The **vision-blind gap** (score with the video minus blind
score) then widens; the paper treats that widening as a secondary, model-dependent check.

To report it:

1. Evaluate the model with video and blind, on `full` and on `debiased`. How to run blind depends
   on the model wrapper; the prompt and decoding should stay the same.
2. Report vision, blind and gap on both sets (micro averages), leading with the change in blind score.
3. Compare with random removal: drop the same number of questions at random (VSI-Bench-Debiased drops 54%) over
   several seeds and report the gap's mean and spread. This shows whether the effect depends on
   which questions were removed; like the gap itself, it is a secondary, model-dependent check.
   `reproduce_v1.py` shows one way to do this with 10 seeds.
4. Parse NUM answers the same way for every model and both arms, and say which parser you used.
   lmms-eval versions differ (older ones take the first word and drop a trailing period; newer
   ones also parse number words), and some models answer with units, words or trailing
   punctuation. Table 9 of the paper parses every NUM answer with
   [`reproduce/robust_num.py`](../reproduce/robust_num.py).

## Responsible use

VSI-Bench and VSI-Bench-Debiased are evaluation sets: do not train or tune on them. For
in-distribution training data, use
[`nyu-visionx/VSI-Train-10k`](https://hf.co/datasets/nyu-visionx/VSI-Train-10k), built from the
training splits of ScanNet, ScanNet++ and ARKitScenes.
