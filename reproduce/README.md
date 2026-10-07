# No-GPU reproduction

This directory re-derives the paper's evaluation numbers that do not need a GPU from released
per-question data: the VSI-Bench-Debiased v1 evaluation (Tables 2, 3, 9, 11 and 12), the automated
IBP results (Tables 8, 10, 13 and 14) and the GPT-4o blind baselines (Table 1). None of it needs a
GPU, model downloads or API keys. Two steps download data from Hugging Face on first use: the
GPT-4o check fetches Video-MME's annotations (about 0.4 MB; Video-MME's gold answers cannot be
redistributed), and `regenerate_ibp.sh` fetches VSI-Bench. `blind_baselines.py --check --benchmarks
vsi cvb mmmu` is fully offline.

```bash
python reproduce/reproduce_v1.py --check       # v1 evaluation: 101 values; numpy and text2digits only
python reproduce/ibp_tables.py --check         # IBP tables: 188 values; numpy only
python scripts/blind_baselines.py --check      # GPT-4o: 4 values; needs the package (and network once)
```

Each `--check` compares with its `expected_*.json` (tolerance 0.005 points), prints one `OK` line
and exits with status 1 on any mismatch.

## VSI-Bench-Debiased v1 evaluation

`reproduce_v1.py` re-derives the v1 evaluation numbers from cached per-sample model predictions
and the released removal list, in about a second. Its only dependencies are numpy and
text2digits, so it also works outside the package environment:

```bash
pip install numpy text2digits   # tested with numpy 1.26.4 and 2.5.3, text2digits 0.1.0
python reproduce/reproduce_v1.py --check   # ends with: OK: all 101 expected values match (tol 0.005).
```

### What it reproduces

Table numbers 1–16 are the same in the COLM 2026 proceedings and the arXiv version.

| Paper | Values printed |
|---|---|
| Tables 2 and 3, fine-tuned LLaVA-Video-7B row | vision 57.05 / blind 44.69 / gap 12.36 on the original set; 48.66 / 32.01 / 16.65 on v1 |
| Tables 2 and 3, base LLaVA-Video-7B blind cells | 25.90 / 20.27 |
| Table 9, v1 composition | 2,768 of 5,130 removed, per question type |
| Table 11, matched random pruning (54%, 10 seeds) | LLaVA-Video FT 12.45 ± 0.61 vs. guided 16.65; Cambrian-S 19.36 ± 0.81 vs. guided 26.04 |
| Table 12, first two rows | Cambrian-S 66.05 / 46.86 / 19.20 on the original set, 58.34 / 32.30 / 26.04 on v1; LLaVA-Video FT as above |
| Table 12, rows 3–7 (vision / blind / gap, original → v1) | InternVL3-9B 43.58 / 32.53 / 11.05 → 41.49 / 28.24 / 13.24; InternVL2.5-26B 45.30 / 32.45 / 12.84 → 38.39 / 25.07 / 13.31; InternVL2.5-8B 44.04 / 31.61 / 12.43 → 37.54 / 24.24 / 13.30; LLaVA-OneVision-7B 35.92 / 26.97 / 8.95 → 29.62 / 21.16 / 8.45; base LLaVA-Video-7B 37.04 / 25.89 / 11.15 → 31.85 / 20.23 / 11.62 |

The unrounded v1 gap of the fine-tuned LLaVA-Video-7B is 16.65. The arXiv version prints it as
16.7 throughout; the proceedings print 16.6 in Table 2 and 16.7 in Tables 10, 11 and 12.

Table 12's base LLaVA-Video-7B row is a 2026 evaluation of the public checkpoint; Tables 2
and 3 report the original 2025 evaluation of the same model. Their blind runs give the same
predictions for 5,103 of 5,130 questions and the same scores (25.90 vs 25.89); only the vision
runs differ.

Scores are micro averages over questions: MC questions use the logged `accuracy`, and every
NUM answer (all models, vision and blind) is re-scored from its `prediction` with
`robust_num.py`: number words become digits, the first number is taken, and lengths in other
units are converted to the unit the question asks for. The records also carry the per-sample
`MRA:.5:.95:.05` field logged with the predictions, which parsed the whole answer with a bare
`float()` and so scored answers such as "0.6." or "1.5 meters." as 0. The two agree for every
run except the base LLaVA-Video-7B and LLaVA-OneVision-7B blind runs, whose answers all end
with a period (the field gives 23.60 / 18.60 and 25.58 / 19.04 instead). NUM parsing differs
across lmms-eval versions, so compare models only under one parser. lmms-eval's `vsibench`
task reports macro averages over question types, so its headline numbers differ. Samples are joined
on the Hugging Face `id` field. The `full` config's row order changed on
2025-11-11, so never join on row position.

### What it does not reproduce

- **Base LLaVA-Video-7B vision cells (36.7 / 31.3).** The cached vision run was made on a
  pre-release version of VSI-Bench and is not shipped. The gap and FT-increase cells of
  Table 2 that use these cells are not reproduced either.
- **Regenerating the predictions.** The Cambrian-S predictions come from a
  pre-release checkpoint, so they differ from the public Cambrian-S-7B.
  The VSI-Train-10k fine-tuned LLaVA-Video-7B checkpoint is not public. The Table 12
  rows 3–7 models are public checkpoints; rerunning them needs a GPU and gives a new run,
  not a byte-level check. This directory re-aggregates the released predictions only.
- **Rebuilding v1 itself.** v1 was built with hand-written per-type filters and hand-set budgets, not by the
  automated IBP algorithm. The released removal list is the canonical artifact.
  `legacy/` holds the script that produced it (see `legacy/README.md`).

### Inputs

- `data/vsi_bench_debiased_v1_removed_ids.txt` lists the 2,768 removed IDs. It is
  byte-identical to `pruned_ids.txt` in
  [`nyu-visionx/VSI-Bench`](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)
  at revision `bdcadb3fea447621a828a24911801faba3587c12` (sha256
  `76fc941f75956f7d3796ae017d2ec06a7eb4c447862624ca27e09c9eb58ae9e9`).
- `data/preds/*.jsonl.gz` are per-sample lmms-eval outputs on VSI-Bench (5,130
  questions each), reduced to `id`, `dataset`, `scene_name`, `question_type`,
  `ground_truth`, `prediction` and the score fields. They contain no question text.
  The `ground_truth` column repeats the answers that VSI-Bench already publishes on
  Hugging Face.

  | File | Model | Input |
  |---|---|---|
  | `vsi_train_10k.jsonl.gz` | LLaVA-Video-7B fine-tuned on VSI-Train-10k | video |
  | `vsi_train_10k_blind.jsonl.gz` | LLaVA-Video-7B fine-tuned on VSI-Train-10k | blind (text only) |
  | `cambrian-s.jsonl.gz` | Cambrian-S, pre-release checkpoint | video |
  | `cambrian-s_blind.jsonl.gz` | Cambrian-S, pre-release checkpoint | blind (text only) |
  | `llava_vid_7b_blind.jsonl.gz` | LLaVA-Video-7B (base), 2025 evaluation | blind (text only) |
  | `internvl3_9b.jsonl.gz`, `internvl3_9b_blind.jsonl.gz` | InternVL3-9B (`OpenGVLab/InternVL3-9B` @ `5f61851`) | video / blind |
  | `internvl2_5_26b.jsonl.gz`, `internvl2_5_26b_blind.jsonl.gz` | InternVL2.5-26B (`OpenGVLab/InternVL2_5-26B` @ `b537a99`) | video / blind |
  | `internvl2_5_8b.jsonl.gz`, `internvl2_5_8b_blind.jsonl.gz` | InternVL2.5-8B (`OpenGVLab/InternVL2_5-8B` @ `e9e4c0d`) | video / blind |
  | `llava_ov_7b.jsonl.gz`, `llava_ov_7b_blind.jsonl.gz` | LLaVA-OneVision-7B (`lmms-lab/llava-onevision-qwen2-7b-ov`) | video / blind |
  | `llava_vid_7b_2026.jsonl.gz`, `llava_vid_7b_2026_blind.jsonl.gz` | LLaVA-Video-7B (base, `lmms-lab/LLaVA-Video-7B-Qwen2`), 2026 evaluation | video / blind |

  The Table 12 rows 3–7 runs use 32 video frames.

- `data/SHA256SUMS` gives the hashes of the original, unreduced lmms-eval logs that
  the reduced files were derived from. They are provenance records; the original
  logs are not shipped, so `sha256sum -c` does not apply to this directory.

The random-pruning control draws with `numpy.random.default_rng(seed)` over the
string-sorted IDs. A re-implementation that orders IDs differently gets a
statistically equivalent result, not an identical one.

### Command-line options

```
python reproduce/reproduce_v1.py [--check [EXPECTED_JSON]] [--json OUT]
                                 [--preds-dir DIR] [--removed-ids FILE] [--tol PTS]
```

`--removed-ids` lets you re-score the same predictions against any other removal
list, for example one you produced yourself.

## IBP and GPT-4o blind baselines

`ibp_tables.py --check` compares 188 values with `expected_ibp.json` (tolerance 0.005) and ends
with `OK: all 188 expected values match (tol 0.005).` It reuses `load_scores` and `random_control`
from `reproduce_v1.py`.

| Paper (arXiv v2) | Values printed |
|---|---|
| Table 10, VSI refinement provenance | fine-tuned LLaVA-Video-7B gap 12.36 on the original set; automated per-format B=1000: 19.5% removed, 4,130 kept, 12.36 (random at 19.5%: mean 12.20, min–max over 10 seeds [11.96, 12.57]); per-type uniform: 54.0%, 2,360 kept, 14.09 (random 12.24); per-type with the pilot's budgets: 2,362 kept, 14.50 (seed 1) / 14.69 (seed 42) (random 12.45); manual pilot v1 16.65. Also checked (code-release values, not in the table): Cambrian-S per-format 20.16, per-type uniform 22.83 |
| Table 13, TsT-LLM vs. TsT-RF removal | Jaccard 0.060 (68 shared of 1,132), 0.050 at a matched budget of 200; removal rates for the four question types in the table (the other six are checked as code-release values) |
| Table 14, per-format budget sweep | B = 200 ... 2,500: 3.9 ... 48.7% removed; mean TsT-RF s(x) over the remaining MC or NUM questions (computed at the start of the last iteration), MC 0.337 ... 0.200, NUM 0.421 ... 0.138 |
| Table 8, MMMU negative control | LLaVA-OneVision-7B 49.4 / 42.8 / 6.67 (900 questions) and 47.9 / 40.4 / 7.50 (800 kept); 1,000 random removals 6.69 ± 0.58, range 4.62 to 8.75 |
| Table 1, GPT-4o column | see "GPT-4o blind baselines" below |

### IBP inputs (`data/ibp/`)

All lists hold question ids (the benchmark's id column), one per line; no question text.

- `rf/<run>/removed_ids.txt` and `rf/<run>/summary.json`: TsT-RF IBP runs written by
  `python -m TsT.debiasing` (VSI-Bench @ `bc96b17`, leak-free `default` features, ties broken by
  question id, seed 42 unless noted). `summary.json` holds the settings, the row-order fingerprint
  and the per-iteration bias traces (Table 14 reads `mean_bias_last_iteration`).

  | Run | Arguments |
  |---|---|
  | `pf_b200` | `--alloc per_format --budget 200 --batch_size 25` |
  | `pf_b500`, `pf_b1000`, `pf_b1500`, `pf_b2000`, `pf_b2500` | `--alloc per_format --budget B` |
  | `pt_uniform54` | `--alloc per_type --frac 0.54` |
  | `pt_v1budgets_s42`, `pt_v1budgets_s1` | `--alloc per_type --budgets_from data/vsi_bench_debiased_v1_removed_ids.txt [--random_state 1]` |

  The B=200 run uses batch size 25, as in the paper's sweep; the others use 50.
  `bash reproduce/regenerate_ibp.sh [OUT_DIR] [JOBS]` reruns all nine (on a 96-core machine, about
  30 minutes one at a time or 10 minutes with JOBS=5) and checks every list against the shipped
  one; it uses `python` from the environment (`uv run bash ...`, or set `PYTHON`). Exact agreement needs the locked numpy / pandas / scikit-learn versions.
- `llm/vsi_pf_mc_removed_ids.txt`, `llm/vsi_pf_num_removed_ids.txt`: TsT-LLM IBP on VSI-Bench
  (Qwen2-7B-Instruct, the App. D.1 settings), ranked by Δs(x), 100 questions per answer format in
  batches of 50, on VSI-Bench @ `d7cb1a3` (the TsT-LLM revision). GPU; the equivalent command is
  `python -m TsT.debiasing --benchmark vsi --mode llm --alloc per_format --budgets mc=100,num=100`.
- `llm/mmmu_delta_removed_ids.txt`: TsT-LLM IBP on the MMMU validation split (@ `364f2e2`), ranked by
  Δs(x): 100 of the 847 multiple-choice questions removed in batches of 20 (the 53 open-ended
  questions are not scored and are kept). GPU; `python -m TsT.debiasing --benchmark mmmu --mode llm
  --alloc global --budget 100 --batch_size 20 --early_stop 0.05` (the early stop never triggered).
- `mmmu/llava_ov_7b_mmmu_val_{vision,blind}.jsonl.gz`: LLaVA-OneVision-7B
  (`llava-onevision-qwen2-7b-ov`) on lmms-eval's `mmmu_val`, with images and with the images withheld,
  reduced to `id`, `question_type` and `correct` (lmms-eval's MMMU judge: official multiple-choice
  parsing and open-ended matching). Table 8 scores are accuracy averaged over questions.

The TsT-LLM lists cannot be regenerated on CPU; a GPU rerun with the commands above is a new run,
not a byte-level check of the shipped lists.

Random controls: VSI-Bench uses `numpy.random.default_rng(seed)` for seeds 0-9 over the
string-sorted ids, as in `reproduce_v1.py`; MMMU uses Python's `random.Random(seed).sample` for
seeds 0-999 over the sorted ids and reports the population standard deviation.

The proceedings' automated IBP numbers were computed on VSI-Bench @ `d7cb1a3` with the `paper`
feature set and pandas' unstable sort. `--revision d7cb1a3 --feature_set paper --tie_break legacy`
re-derives those removal sets (checked for per-format B=1000) on x86-64 CPUs where numpy 1.26.4 uses
its AVX-512 sort; elsewhere, including Apple Silicon, ties come out in another order and about three
quarters of that set matches. The v2 numbers above use the pinned revision, leak-free features and
the id tie-break, which give the same lists on every platform tested.

### GPT-4o blind baselines (`data/gpt4o/`)

`<benchmark>_mc.jsonl.gz` holds GPT-4o's (`gpt-4o-2024-08-06`, temperature 0, at most 10 output
tokens) answers to the blind prompts of every multiple-choice question: `id`, `ground_truth`
(letter), `n_options`, `prediction` (the first token of the response, which is what gets scored)
and `api_error` (the request failed; scored as unparseable). No question or option text is
included. VSI-Bench responses were collected on the 2025-11-11 revision (`d7cb1a3`), the others on
the pinned revisions; the ids were matched to each question's text and gold answer.
`python scripts/blind_baselines.py --run` queries the API again (your key, your cost).

A response that names no option (mostly "I'm sorry, I can't see images") counts as incorrect, as
in the paper, as does a failed request. That rule decides 5 VSI-Bench, 232 CV-Bench, 7 MMMU and 8
VideoMME questions (including 5 VSI-Bench and 1 MMMU failed requests).
`--unparseable random` gives them a random option instead, as the MMMU answer parser does, seeded
by (seed, benchmark, question id):

| Benchmark | Paper (arXiv Table 1) | Default (unparseable = incorrect) | `--unparseable random` |
|---|---:|---:|---:|
| VSI-Bench MC (2,490) | 34.0 | 33.98 | 34.02 |
| CV-Bench (2,638) | 44.8 | 44.84 | 47.69 |
| MMMU MC (847) | 52.3 | 52.30 | 52.66 |
| VideoMME (2,700) | 46.6 | 46.59 | 46.63 |
