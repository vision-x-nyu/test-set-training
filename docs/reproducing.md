# Reproducing the paper

This page lists every paper number the release reproduces, the command that does it, and what to
expect. Numbers follow the corrected [arXiv version](https://arxiv.org/abs/2511.04655); its
Appendix G lists what changed since the COLM proceedings.

## At a glance

| Paper | Command | Runs on |
|---|---|---|
| Table 4 (TsT-RF) | `uv run python -m TsT -b vsi` (and `-b cvb`, `--group_col ...`) | CPU, seconds |
| Tables 2, 3, 9, 11, 12 (VSI-Bench-Debiased v1 evaluation) | `uv run python reproduce/reproduce_v1.py --check` | CPU, 1 second |
| Tables 8, 10, 13, 14 (automated IBP) | `uv run python reproduce/ibp_tables.py --check` | CPU, 1 second |
| Table 1, GPT-4o column | `uv run python scripts/blind_baselines.py --check` | CPU, no API calls (downloads Video-MME annotations once) |
| App. F and Table 15 (size case study) | `uv run python -m TsT -b vsi -q object_size_estimation --verbose` | CPU, seconds |
| Table 1 (TsT-LLM) | `uv run python -m TsT -b <benchmark> --mode llm --output_dir outputs/<benchmark>_llm` | one GPU, 20–50 minutes each |

Every CPU value reproduces exactly with the locked versions, and `reproduce/regenerate_ibp.sh`
reruns IBP to regenerate the shipped removal lists byte for byte. LoRA training on GPUs is not
bit-reproducible, so TsT-LLM reruns land within the run-to-run variation App. D.1 describes. The
default TsT-RF features are leak-free; `--feature_set paper` reproduces the proceedings' VSI-Bench
43.5 and warns that it reads held-out labels, and the proceedings' CV-Bench 75.5 is withdrawn.

## Conventions

Expected values are weighted means over questions, in points, at the pinned dataset revisions.
Table numbers 1–16 are the same in the COLM 2026 proceedings and the
[arXiv version](https://arxiv.org/abs/2511.04655); where the two print different values, the row
says which one it reproduces. **Exact** means the command reproduces, to the precision printed, a
value printed in the cell named in the row; a value that no paper version prints is marked
**code-release value (not reported in the paper)**. Not scripted here: Figures 2 and 3 (blind
scores across LLaVA-OneVision sizes, answer distributions), the App. D.3 forest-size grid, and the
App. D.1 and E.3 comparisons with a chat-formatted configuration and with LLaMA-3-8B as the
diagnostic model.

## TsT-RF and the no-GPU reproductions

These run on a CPU and are deterministic with the locked versions.

| Paper | Command | Expected | Status |
|---|---|---|---|
| Table 4 (arXiv), VSI-Bench, random folds: 43.0 | `uv run python -m TsT -b vsi` | 43.00 | Exact |
| Table 4 (arXiv), VSI-Bench, grouped folds: 38.8 | `uv run python -m TsT -b vsi --group_col scene_name` | 38.82 | Exact |
| Table 4 (arXiv), CV-Bench, random folds: 56.1 | `uv run python -m TsT -b cvb` | 56.14 | Exact. The proceedings' 75.5 is withdrawn |
| Table 4 (arXiv), CV-Bench, grouped folds: 55.7 | `uv run python -m TsT -b cvb --group_col image_id` | 55.65 | Exact |
| Table 4 of the proceedings, VSI-Bench: 43.5 | `uv run python -m TsT -b vsi --feature_set paper` | 43.50 | Exact. This preset reads the held-out gold pair and prints a warning (see [the notes](#notes-on-this-code-and-the-paper)) |
| — | `uv run python -m TsT -b vsi --feature_set paper --group_col scene_name` | 38.98 | Code-release value (not reported in the paper) |
| App. F and Table 15, VSI-Bench size estimation: 61.4% MRA; `obj_val_log_mean` importance 0.968 | `uv run python -m TsT -b vsi -q object_size_estimation --verbose` | 61.35; 0.968 | Exact |
| Tables 2 and 3 (except the base LLaVA-Video-7B vision cells and the cells derived from them), Table 9, Table 11, Table 12 (arXiv) | `uv run python reproduce/reproduce_v1.py --check` | 101/101 values match, e.g. fine-tuned LLaVA-Video-7B blind 44.69 → 32.01 and gap 12.36 → 16.65; matched random removal 12.45 ± 0.61 | Exact ([details](../reproduce/README.md)) |
| Tables 8, 10, 13 and 14 (arXiv: MMMU negative control, automated IBP, diagnostic overlap, budget sweep) | `uv run python reproduce/ibp_tables.py --check` | 188/188 values match, e.g. per-format IBP gap 12.36 (random 12.20), per-type 14.09 (random 12.24), Jaccard 0.060 | Exact, from the shipped removal lists ([details](../reproduce/README.md#ibp-and-gpt-4o-blind-baselines); the proceedings' selections are covered there too) |
| The TsT-RF removal lists behind Tables 10, 13 and 14 | `uv run bash reproduce/regenerate_ibp.sh outputs/ibp 5` | the 9 shipped lists, regenerated with `python -m TsT.debiasing` | Exact (about 10 minutes with 5 parallel runs; 30 minutes one at a time) |
| Table 1 (arXiv), GPT-4o column: VSI-Bench MC 34.0, CV-Bench 44.8, MMMU 52.3, VideoMME 46.6 | `uv run python scripts/blind_baselines.py --check` | 33.98, 44.84, 52.30, 46.59 | Exact, from the shipped GPT-4o responses (no API calls) |
| VSI-Bench-Debiased v1 | [`reproduce/data/vsi_bench_debiased_v1_removed_ids.txt`](../reproduce/data/vsi_bench_debiased_v1_removed_ids.txt) | 2,768 IDs, the same set as `pruned_ids.txt` on Hugging Face | The released list is canonical. The script that built it is in [`legacy/`](../legacy/README.md) |
| Tables 2 and 3, base LLaVA-Video-7B vision cells | — | — | Not reproducible: the cached vision run was made on a pre-release version of VSI-Bench and is not shipped |

## TsT-LLM

Table 1; one GPU and the `llm` extra ([tst-llm.md](tst-llm.md)); the base model is pinned to the
revision of the paper's runs. LoRA training on GPUs is not bit-reproducible, so the "Release run" column gives one run of this code
(zero-shot → TsT, ΔTsT); these values are code-release values. The paper's zero-shot scores also
carried up to a few tenths of a point of noise from an unseeded fallback that this release seeds.

| Paper (arXiv Table 1): zero-shot → TsT (ΔTsT) | Command | Release run |
|---|---|---|
| VSI-Bench (MC+NUM): 24.7 → 42.7 (+17.9); MC 25.5 → 39.1 (+13.6); NUM 24.0 → 46.0 (+21.9) | `uv run python -m TsT -b vsi --mode llm --output_dir outputs/vsi_llm` | 24.66 → 42.79 (+18.13); MC 25.43 → 39.34 (+13.91); NUM 23.94 → 46.04 (+22.10) |
| CV-Bench: 47.3 → 60.3 (+13.1) | `uv run python -m TsT -b cvb --mode llm --output_dir outputs/cvb_llm` | 47.06 → 61.09 (+14.04). App. D.1: CV-Bench's gain varies by several points across reruns and fold assignments |
| VideoMME: 29.2 → 33.4 (+4.1) | `uv run python -m TsT -b video_mme --mode llm --output_dir outputs/video_mme_llm` | 29.23 → 33.37 (+4.14) |
| MMMU (validation, MC): 34.2 → 33.6 (−0.7); top-1 accuracy +5.4 (Sec. 3.4) | `uv run python -m TsT -b mmmu --mode llm --output_dir outputs/mmmu_llm` | 34.36 → 33.64 (−0.72); top-1 accuracy +5.08 |
| MMStar: 24.2 → 29.5 (+5.4) | `uv run python -m TsT -b mmstar --mode llm --output_dir outputs/mmstar_llm` | 23.97 → 29.46 (+5.49) |

Top-1 accuracy comes from `uv run python scripts/summarize_llm_predictions.py outputs/mmmu_llm/predictions`.

## Notes on this code and the paper

Appendix G of the arXiv version (version history) lists the main changes to the paper's numbers
since the proceedings; this page does not restate them. (The automated IBP numbers in Tables 10,
13 and 14 also changed, by up to 0.7 points in Table 10 and about 2 points in Table 13, when rerun at
the pinned revision with leak-free features and ties broken by question id; see
[reproduce/README.md](../reproduce/README.md#ibp-and-gpt-4o-blind-baselines).) Notes about the code:

- **Feature presets.** `--feature_set default` (recommended) is leak-free: no feature reads a
  held-out question's own answer. For VSI-Bench it drops the 10 relative-distance columns that
  Table 6 of the arXiv version marks as looked up with the held-out question's own gold object
  (`opt_{0..3}_tgt_option_pair_freq`, `opt_{0..3}_tgt_option_ord_pair_freq`,
  `max_tgt_option_pair_freq`, `max_tgt_option_ord_pair_freq`). `--feature_set paper` keeps them
  only to reproduce the proceedings' 43.50, and warns on every run. CV-Bench has only the default
  preset; its target is the gold option index. Some columns have different names in the paper's
  feature tables; [interpreting-results.md](interpreting-results.md#feature-names-in-the-paper-and-in-the-code)
  maps them.
- **Dataset revisions and row order.** Shuffled k-fold assignment depends on row order, so each
  benchmark is pinned to a Hugging Face revision; `--revision` overrides it. The loader reads only
  the requested revision's files, also offline, and checks every load against the revision's
  row-order fingerprint (`row_order_sha1` in `summary.json`: SHA-1 of the comma-joined question
  ids in row order). A mismatch stops the run with an error instead of printing a score; a
  revision other than the pinned one also prints the fingerprint it loaded.

  | Benchmark | Dataset | Pinned revision | Row-order fingerprint | Note |
  |---|---|---|---|---|
  | VSI-Bench | [`nyu-visionx/VSI-Bench`](https://hf.co/datasets/nyu-visionx/VSI-Bench) | `bc96b17` (TsT-RF); `d7cb1a3` (TsT-LLM, as in the paper's runs) | `8c48d42738671ef11a1ccb6300c81ed5d893f10f` | The row order of the proceedings' TsT-RF runs. Later revisions (from 2025-11-11, e.g. `d7cb1a3`, fingerprint `b48270fe65bfe142baa12835a230797f6e4f4ec7`) list the same questions in another order; there the paper preset gives 43.64 instead of 43.50. |
  | CV-Bench | [`nyu-visionx/CV-Bench`](https://hf.co/datasets/nyu-visionx/CV-Bench) | `bc284db` | `02535f1c609db8bba7c6bd64d9a78b9aa3d0fa3a` | Fingerprint over the `idx` column |
  | VSI-Bench-Debiased v1 IDs | [`nyu-visionx/VSI-Bench`](https://hf.co/datasets/nyu-visionx/VSI-Bench) | `bdcadb3` | — | `pruned_ids.txt`; matches [`reproduce/data/`](../reproduce/data) |

- **MRA threshold.** TsT scores NUM questions with mean relative accuracy over thresholds
  0.50, 0.55, …, 0.95 using a strict `<` (relative error < 1 − threshold). The official VSI-Bench
  evaluation (and lmms-eval) uses `<=`. The two differ when a prediction's relative error lands
  exactly on a threshold, which happens for some integer answers (for example counts).
- **TsT-LLM, as run.** Training formats examples with LLaMA-Factory's `gemma` template while
  inference sends raw prompts, as App. D.1 states; the defaults keep this so the paper's numbers
  reproduce. When a multiple-choice question's gold letter is not among the top-100
  log-probabilities, the generated text is parsed, and text that names no option gets a random
  option; this release seeds that choice per question, while the paper's runs drew it unseeded
  (up to a few tenths of a point of run-to-run noise in zero-shot scores). A numerical answer with no number
  scores 0. GPU training is not bit-reproducible, so reruns differ slightly fold by fold.
- **IBP.** Candidates with equal scores are ordered by question id, so given the scores the ranking
  does not depend on row order or sort stability. The scores themselves depend on row order through
  the shuffled folds, which is why datasets are pinned by revision and row-order fingerprint. The
  paper's selections reproduce exactly
  (`reproduce/regenerate_ibp.sh`); the 200-question budget of Table 14 used batches of 25 instead
  of the default 50.
