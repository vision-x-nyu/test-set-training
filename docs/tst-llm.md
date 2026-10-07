# TsT-LLM

TsT-LLM is the paper's primary diagnostic. For each of k folds, it LoRA-fine-tunes a text-only
LLM on the other folds' questions and answers and scores the held-out fold; a zero-shot pass of the
base model over every question gives the baseline. The gain over that baseline, ΔTsT, is what the
model learned from the test set's own text. The defaults are the paper's settings (App. D.1); see
[paper-settings.md](paper-settings.md) for every value.

## Requirements

- Linux x86_64 with one NVIDIA GPU and a driver for CUDA 12.8 (driver 570 or newer). The paper's settings
  (Qwen2-7B-Instruct, LoRA, batch size 4, fp16) were run on data-center GPUs with 80 GB or more;
  vLLM reserves 80% of the GPU's memory for inference.
- The `llm` extra: vLLM 0.9.2, PyTorch 2.7.0 with CUDA 12.8 wheels, transformers 4.52.4 and
  LLaMA-Factory at upstream commit `b8272a8`:

  ```bash
  uv sync --frozen --extra llm
  ```

  Use uv for the `llm` extra: `uv.lock` pins every package, including the CUDA wheel index
  for PyTorch and the LLaMA-Factory commit, which is installed from GitHub (so `git` and access
  to github.com are needed). The extra adds about 10 GB of packages. (`constraints.txt` covers the
  CPU core only.)
- Network access to the Hugging Face Hub on first use: about 15 GB for Qwen2-7B-Instruct, plus
  the benchmark files (MMMU's validation file embeds images and is about 340 MB).

## Run it

```bash
uv run python -m TsT --benchmark vsi --mode llm --output_dir outputs/vsi_llm
```

The run prints the zero-shot and TsT scores for each answer format (`vsi_mc`, `vsi_num`) and
overall:

```
question type format metric zero-shot  score   std    n
       vsi_mc     mc    acc     ...      ...   ...  2490
      vsi_num    num    mra     ...      ...   ...  2640

weighted mean: ...   macro mean: ...   (questions scored: 5130)
zero-shot weighted mean: ...   delta (TsT - zero-shot), weighted: ...
```

For MC questions, "acc" is the mean probability of the gold option (a soft accuracy); for NUM
questions it is the mean relative accuracy (MRA) of the generated number.

Each fold trains a fresh adapter, so a run makes k + 1 passes over the benchmark per answer
format. On one 80 GB data-center GPU, VSI-Bench takes 40–50 minutes and CV-Bench, VideoMME, MMMU
or MMStar 20–30 minutes. Each fold's adapter is written to a temporary directory (`$TMPDIR`) and
deleted once the fold is scored.

| Benchmark | `--benchmark` | Questions TsT-LLM scores | Dataset revision |
|---|---|---|---|
| VSI-Bench | `vsi` | 5,130 (2,490 MC, 2,640 NUM) | `d7cb1a3` (TsT-RF uses `bc96b17`) |
| CV-Bench | `cvb` | 2,638 MC | `bc284db` |
| VideoMME | `video_mme` | 2,700 MC | `ead1408` |
| MMMU (validation) | `mmmu` | 847 MC (the 53 open-ended questions are not scored) | `364f2e2` |
| MMStar | `mmstar` | 1,500 MC | `bc98d66` |

The common options (`--n_splits`, `--repeats`, `--random_state`, `--fold_seed`, `--group_col`,
`--revision`, `--output_dir`, `--keep_going`) work as for TsT-RF; see
[getting-started.md](getting-started.md). TsT-LLM options:

| Option | Default | Meaning |
|---|---|---|
| `--llm_model` | `Qwen/Qwen2-7B-Instruct` | Base model. Training formats examples with LLaMA-Factory's `gemma` template, the paper's setting for Qwen2; for other model families set `LLMRunConfig(template=...)` to their LLaMA-Factory template (e.g. `llama3`, `mistral`, `qwen`), since `gemma`'s stop word is not in every tokenizer and training then fails |
| `--llm_model_revision` | see meaning | Base-model revision. For Qwen2-7B-Instruct the default is the paper's `f2826a00ceef68f0f2b946d945ecc0477ce4450c`; for other models, the Hub's `main` |
| `--llm_train_batch_size` | 4 | LoRA training batch size |
| `--llm_eval_batch_size` | 4 | vLLM batch size |
| `--llm_epochs` | 1 | LoRA epochs per fold |

Other settings (LoRA rank and alpha, learning rate, sequence length, template) are fields of
`TsT.evaluators.llm.LLMRunConfig`, which the Python API accepts:

```python
from TsT import evaluate_benchmark
from TsT.evaluators.llm import LLMRunConfig

if __name__ == "__main__":  # vLLM may start its engine in a spawned process, which re-imports this file
    run = evaluate_benchmark("cvb", mode="llm", llm_config=LLMRunConfig(lora_rank=16, lora_alpha=32),
                             predictions_dir="outputs/cvb_llm/predictions")
    print(run.summary()["delta_weighted"])
```

## Outputs

With `--output_dir DIR`:

- `DIR/summary.json`: as for TsT-RF, plus `mode: "llm"`, the `llm_config`, each answer format's
  `zero_shot` score, `zero_shot_weighted_mean`, `delta_weighted` (ΔTsT over all questions) and the
  versions of PyTorch, vLLM, transformers, PEFT, TRL and LLaMA-Factory.
- `DIR/s_x.csv`: one row per question with `s` (its held-out score), `s_zero_shot` (the base
  model's score) and `delta` (`s - s_zero_shot`, the per-question gain that IBP ranks by when it
  uses TsT-LLM scores).
- `DIR/predictions/`: one JSON-lines file per zero-shot pass and fold
  (`vsi_mc_zero_shot.jsonl`, `vsi_mc_seed42_fold1.jsonl`, ...), with each question's id, gold
  answer, options, generated answer, option probabilities and score.
  `scripts/summarize_llm_predictions.py DIR/predictions` recomputes the scores from these files
  and adds hard top-1 accuracy (the most probable option letter), which the paper quotes for MMMU.
  These files contain each question's gold answer. Video-MME does not allow redistributing any
  part of it and MMStar declares no license, so do not publish prediction files for those
  benchmarks.

## Notes

- **Scoring.** MC questions are scored by the probability of the gold option letter among the
  top-100 log-probabilities of the first generated token, normalized over the option letters
  found. If the gold letter is not among them, the generated text is parsed instead (score 0 or
  1), and text that names no option gets a random option. That random choice is seeded by the
  run seed and the question, so it no longer varies between reruns; the paper's runs drew it
  unseeded, which moved zero-shot scores by up to a few tenths of a point between runs. NUM answers with no number
  score 0.
- **Fold assignment.** On CV-Bench, ΔTsT moves by up to about 5 points with the fold split alone
  (`--fold_seed`); don't pick among fold seeds, and report the range over a few seeds when
  compute allows.
- **Grouped folds.** `--group_col scene_name` (VSI-Bench), `image_id` (CV-Bench) or `videoID`
  (VideoMME) keeps all questions about one scene, image or video in one fold. With scene-grouped
  folds, the paper reports VSI-Bench's ΔTsT as +17.0 instead of +17.9.
- **One GPU.** The trainer exposes only the first visible GPU to LoRA training, as in the paper's
  runs; set `CUDA_VISIBLE_DEVICES` to choose it.
- **Several runs at once.** vLLM caches compiled models under `~/.cache/vllm`. Runs that start
  at the same time and share that directory (for example, cluster jobs with a shared home
  directory) can corrupt it, and vLLM then fails at start-up with "failed to compile the model".
  Give each run its own cache, e.g. `export VLLM_CACHE_ROOT="$(mktemp -d)"`.
