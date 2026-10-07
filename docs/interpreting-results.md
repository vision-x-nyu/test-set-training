# Interpreting TsT results

Most of this page is about TsT-RF; [TsT-LLM scores](#tst-llm-scores) covers what differs for
`--mode llm`.

## What the benchmark-level score measures

Every question is scored by a random forest trained only on the other folds' questions of the same
type, using non-visual features: parsed question text, option properties, and statistics of the
training folds' answers. The held-out score is therefore an estimate of how much of the test set
can be answered from patterns that the test set itself teaches, without the image or video.

- MC question types are scored by accuracy, NUM types by mean relative accuracy (MRA).
- The **weighted mean** counts every question once; the paper reports this number. The **macro
  mean** averages question types equally.
- Compare each question type with its chance level (1 / number of options for MC; a type that
  mixes option counts has a mixed chance level). For NUM types, compare with a constant baseline
  such as the training-fold median or most frequent answer: under MRA the mean is a weak baseline,
  and the forests minimise squared error, not MRA, so they can score below a good constant (on
  VSI-Bench counting, always answering 2 beats TsT-RF).
- Per-type scores, and the random- vs grouped-fold comparison, move by a few points with the fold
  split; use `--repeats` or a `--fold_seed` sweep before reading small differences.

What the score is not:

- **Not a blind model's score.** Many features are statistics of gold answers in the training
  folds, which no deployed model sees. Read TsT-RF as a designer-side audit, not as a baseline
  that a model "should" beat, and do not report it as ordinary no-access model performance.
- **Only a lower bound.** A low score does not show that a shortcut is absent. A forest sees only
  the features someone wrote, so a shortcut no feature captures goes unnoticed; the paper's
  TsT-LLM diagnostic needs no feature engineering, and a stronger diagnostic could find more.
- **Not a measure of pretrained knowledge.** A text-only LLM can answer some questions from world
  knowledge without any test-set pattern. TsT measures what is learnable from the test set; a blind
  score alone cannot separate the two.

## Random versus grouped folds

With question-level folds, a held-out question can share a scene, video or image with training
questions, so features keyed on that shared context (for example, the objects in the same room)
can carry over. Grouped folds (`--group_col scene_name` for VSI-Bench, `--group_col image_id` for
CV-Bench) keep each group in one fold. Report both. A large drop under grouping means much of the
signal is shared context within a scene or image; what survives grouping is closer to a property of
the question templates and answer distributions themselves.

## The per-question score s(x)

`s_x.csv` has one row per question:

| Column | Meaning |
|---|---|
| `id` (VSI-Bench) or `idx` (CV-Bench) | The benchmark's question id, named as in the Hugging Face dataset and joining it one to one (`summary.json` records the name as `id_col`) |
| `model` | The random-forest model that scored the question |
| `question_type` | The dataset's own question type (several can share one model, e.g. the three VSI-Bench relative-direction levels) |
| `format`, `metric` | `mc` with `acc`, or `num` with `mra` |
| `n_options`, `chance` | MC only: the number of options and 1 / `n_options`; empty for NUM |
| `s` | The held-out score: the forest's probability of the gold answer (MC) or the MRA of its prediction (NUM), averaged over repeats |
| `s_minus_chance` | `s - chance` (MC only) |
| `n_repeats` | How many repeats scored the question |

How to use it:

- **Rank within one format, never across formats, and preferably within one question type.** Raw
  P(gold) is higher for questions with fewer options (chance is 0.5 for two options and 0.25 for
  four), and MRA is on a different scale from probabilities. Within MC, sort by `s_minus_chance`;
  within NUM, sort by `s`.
- **A high s(x) flags a question for review, not for removal.** It means the answer is predictable
  from text-side patterns in the rest of the test set. The designer decides whether that pattern is
  a construction artifact (for example, one answer dominating a template) or legitimate world
  knowledge the benchmark is happy to reward.
- **A low s(x) does not mean a question requires vision.** It only means these features, trained on
  this test set, did not predict it.
- **Join on the id column (`id` for VSI-Bench, `idx` for CV-Bench), never on row position.**
  Benchmarks can reorder rows between revisions (VSI-Bench did on 2025-11-11).
- Single-run s(x) values are noisy, especially for small question types. `--repeats` averages over
  several fold splits (`n_repeats` records how many).

## TsT-LLM scores

TsT-LLM scores every question twice: once with the base model (zero-shot) and once with the LoRA
adapter trained on the other folds. For MC questions both scores are the probability the model
gives the gold option letter; for NUM questions both are the MRA of the generated number.

- **ΔTsT** (`delta_weighted` in `summary.json`) is the fine-tuned score minus the zero-shot score,
  over all questions. It is the part of the blind score that the model learned from the test set's
  own text. The zero-shot score alone mixes pretrained knowledge with answer priors. A small ΔTsT
  next to a high blind score points to knowledge (as on MMMU in the paper, where GPT-4o answers
  52.3% of the multiple-choice questions blind), and a large ΔTsT points to learnable test-set patterns (as on VSI-Bench).
- **Per question**, `s_x.csv` has `s` (the held-out fine-tuned score), `s_zero_shot` and `delta`.
  IBP with TsT-LLM ranks by `delta`, so a question the base model already answered does not count
  as a learned shortcut.
- The probability of the gold option is a soft score. `scripts/summarize_llm_predictions.py`
  reports hard top-1 accuracy (the most probable option letter) from the saved predictions; the
  two can disagree in sign when the gain is small (the paper's MMMU: −0.7 on gold-option
  probability, +5.4 in top-1 accuracy).
- Unlike TsT-RF, TsT-LLM reads no gold answers at test time: the adapter learns only from the
  training folds. Its scores still come from a model fit on the test set and are not ordinary
  no-access model performance.

## Feature importances

`--verbose` logs each model's Gini feature importances, from a forest refit on all of that type's
questions, and `summary.json` lists each model's feature columns. Importances show which features
the forest leaned on; Gini importance favours continuous and high-cardinality features, so treat it
as a pointer for inspection, not as an effect size.

## Feature names in the paper and in the code

The paper's feature tables (Tables 5–7 in both the proceedings and the arXiv version) are
representative, not exhaustive, and some of their names differ from the column names in the code
(`summary.json` lists every model's exact `feature_cols`). The rows below are the names that differ;
every other name in those tables is the same in the code, except two CV-Bench features that the
arXiv version's Table 7 marks ‡ (`pair_answer_freq_score`, `is_majority_answer`): they read the
held-out question's own answer and are not in this release. Indices in the code start at 0 (`opt_0`, `seq_0_score`, `choice_0_…`), except in
the `object_*` and `opt_seq_*` columns, which start at 1.

| Table | Paper name | Code name | Preset |
|---|---|---|---|
| Table 6, object relative distance | `object_{i}`, i ∈ [4] | `object_1` … `object_4` | both |
| Table 6, object relative distance | `opt_{i}_obj_freq` | `opt_{i}_option_freq` | both |
| Table 6, object relative distance | `max_opt_obj_freq` | `max_option_freq` | both |
| Table 6, object relative distance | `opt_{i}_pair_freq`, and in the arXiv version also `opt_{i}_ord_pair_freq` | `opt_{i}_tgt_option_pair_freq` and `opt_{i}_tgt_option_ord_pair_freq` (sorted and ordered pair) | `paper` only. The proceedings describe option i's object paired with the target object; that per-option computation was never implemented. All eight columns hold the training-fold frequency of the held-out question's own (target, gold object) pair, so they read the held-out label. The arXiv version's Table 6 describes what was computed and marks these features ‡. The default preset drops them. |
| Table 6, object relative distance | `max_opt_pair_freq`, and in the arXiv version also `max_opt_ord_pair_freq` | `max_tgt_option_pair_freq` and `max_tgt_option_ord_pair_freq` | `paper` only; the maximum of the columns in the row above, so also the gold-pair frequency |
| Table 6, appearance order | `opt_seq_{i}` | `opt_seq_1` … `opt_seq_4` | both |
| Table 6, appearance order | `seq_{i}_adj_pair_score` | `seq_{i}_pair_score` | both |
| Table 7, 2D count | `opt_{i}_dist_from_obj_mean` | `choice_{i}_dist_from_obj_mean` (i = 0…5) | default (CV-Bench's only preset) |
| Table 7, 2D count | `opt_{i}_dist_from_global_mean` | `choice_{i}_dist_from_global_mean` (i = 0…5) | default |
| Table 7, 2D relation, 3D depth, 3D distance | `object_{i}`, i ∈ [2] | `object_1`, `object_2` | default |

## Checks before trusting a number

- The run used the leak-free `default` feature set (`summary.json`: `"feature_set": "default"`).
  The VSI-Bench `paper` preset reads held-out answers and exists only to reproduce the proceedings.
- `summary.json` shows `"complete": true` (no failed question types).
- The dataset revision and `row_order_sha1` match the run you compare against.
- For your own features, the label-invariance check passes ([your-benchmark.md](your-benchmark.md)).
