# Development

This page describes the repository layout and the checks to run when changing the code.

## Environment and checks

```bash
uv sync --frozen --dev
uv run pytest                     # offline unit and release tests (about 2 minutes)
uv run pytest -m network          # golden values against the pinned datasets (downloads about 0.75 GB)
uv run pre-commit run --all-files # ruff lint and format
```

The release checks that matter most for the paper's numbers:

```bash
uv run python reproduce/reproduce_v1.py --check
uv run python reproduce/ibp_tables.py --check
uv run python scripts/blind_baselines.py --check
uv run bash reproduce/regenerate_ibp.sh outputs/ibp 5   # reruns IBP; about 10 minutes
```

TsT-LLM needs a GPU, so its tests replace vLLM and LLaMA-Factory with fakes
(`tests/llm/test_llm_end_to_end.py`); they check prompts, folds, scoring, the training config and
the output files. A GPU run of `python -m TsT --mode llm` on a small model (for example
`--llm_model Qwen/Qwen2-0.5B-Instruct -k 2`) exercises the real training and inference path in
about 15 minutes.

## Code map

```text
src/TsT/
├── __main__.py               # python -m TsT: TsT-RF and TsT-LLM command line
├── evaluation.py             # evaluate_benchmark, k-fold runs, summary.json and s_x.csv
├── utils.py                  # pinned Hugging Face loading, row-order fingerprints, MRA
├── core/
│   ├── benchmark.py          # Benchmark base class and registry (revisions, fingerprints)
│   ├── cross_validation.py   # folds (question-level or grouped), repeats, fold seeds
│   ├── protocols.py          # model, evaluator and result types
│   └── qa_models.py          # question-answer models for TsT-LLM
├── evaluators/
│   ├── rf.py                 # TsT-RF: random forest per question type
│   └── llm/                  # TsT-LLM
│       ├── data/conversion.py    # blind prompts
│       ├── predictors/vllm.py    # vLLM inference, gold-option probabilities
│       ├── trainers/llamafactory.py  # LoRA training with LLaMA-Factory (one GPU)
│       ├── scoring.py            # MC, NUM and open-ended scoring
│       └── evaluator.py          # zero-shot pass and k-fold LoRA loop
├── debiasing/                # IBP: python -m TsT.debiasing
│   ├── ibp.py                # the iterative loop and debias_benchmark
│   ├── allocation.py         # global, per-format and per-type budgets
│   └── strategies.py         # ranking, ties and group floors
└── benchmarks/               # adapters: vsi and cvb (both diagnostics); video_mme, mmmu, mmstar (TsT-LLM)
scripts/                      # rf_report.py, summarize_llm_predictions.py, blind_baselines.py
reproduce/                    # no-GPU reproduction scripts and their data
legacy/                       # the script that built VSI-Bench-Debiased v1
configs/app_d1/               # the TsT-LLM training config the trainer writes
examples/                     # bring-your-own-benchmark example
tests/                        # unit, release and fake-GPU tests; fixtures are small text-only subsets
```

## Conventions

- **Pin data.** A benchmark adapter pins its Hugging Face revision (`default_revision`, and
  `llm_revision` if TsT-LLM used another one) and the row-order fingerprint of each revision it
  supports. Shuffled folds depend on row order, so unpinned data give different numbers.
- **Never read the held-out label.** TsT-RF features may use statistics of the training folds only.
  `tests/release/test_label_invariance.py` checks every VSI-Bench and CV-Bench feature model by
  permuting the held-out fold's answers; extend it when you add a benchmark with TsT-RF models.
- **Fail loudly.** A model that fails stops the run unless `--keep_going` is given, and then the
  summary marks the run incomplete. Do not catch exceptions to keep a score.
- **Keep the paper's computation.** Changes that move a published number need a reason, a test and
  a note in [reproducing.md](reproducing.md); the `--check` scripts catch accidental changes.
- **Keep imports light.** `import TsT` and `python -m TsT --help` must not import torch, vLLM or
  `datasets`; heavy dependencies load inside the code paths that need them.
