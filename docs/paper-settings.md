# Paper settings: TsT-LLM (App. D.1)

The defaults of `python -m TsT --mode llm` are the configuration behind the paper's
TsT-LLM results (App. D.1). This page lists every setting and where the code sets it.

- [`configs/app_d1/llamafactory_train.yaml`](../configs/app_d1/llamafactory_train.yaml)
  is the LLaMA-Factory config that the trainer writes for each fold, with the two
  per-fold paths replaced by placeholders. A test checks that the trainer still renders
  it byte for byte (`tests/llm/test_llm_config_and_prompts.py`).

## Pinned versions

| Component | Pin | Notes |
|---|---|---|
| Base model | [`Qwen/Qwen2-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct) at `f2826a00ceef68f0f2b946d945ecc0477ce4450c` | The paper's runs passed no revision, but the Hub's `main` has pointed at this commit since 2024-08-21; this release pins it by default. |
| Training framework | [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) at `b8272a874b5ca59762d5386afcfd3d2fb71d5e00` | Installed by the `llm` extra. The paper's runs used `53a9924ea882250cf96e499d271b666956f8b12b`, which has the same commit message, date and package source (`src/`, `setup.py`, `pyproject.toml`, `requirements.txt`) but was later dropped from upstream history. `b8272a8` is an ancestor of `main` and of tags `v0.9.4` and `v0.9.5`. |
| Inference and training libraries | vLLM 0.9.2, PyTorch 2.7.0 (CUDA 12.8), transformers 4.52.4, PEFT 0.15.2, TRL 0.9.6 | Pinned in `uv.lock` (`llm` extra). |
| VSI-Bench | [`nyu-visionx/VSI-Bench`](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) at `d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d` | The revision App. D.1 names for TsT-LLM. (TsT-RF in this release defaults to `bc96b17`; see [reproducing.md](reproducing.md).) |
| CV-Bench | [`nyu-visionx/CV-Bench`](https://huggingface.co/datasets/nyu-visionx/CV-Bench) at `bc284db50d036958861cb60cdd7b77612052ce0d` | The revision App. D.1 of the arXiv version names; TsT-RF in this release uses the same one. |
| VideoMME | `lmms-lab/Video-MME` at `ead1408f75b618502df9a1d8e0950166bf0a2a0b` | The revision App. D.1 of the arXiv version names. The Hub now redirects this repository id. |
| MMMU | `lmms-lab/MMMU` at `364f2e2eb107b36e07ff4c5a15f5947a759cef47` | Validation split. The Hub's head since before the paper's runs, which did not pin it; the Hub now redirects this repository id. |
| MMStar | [`Lin-Chen/MMStar`](https://huggingface.co/datasets/Lin-Chen/MMStar) at `bc98d668301da7b14f648724866e57302778ab27` | The revision of the paper's MMStar run. |

## App. D.1, value by value

| App. D.1 setting | Value | Where it is set |
|---|---|---|
| Model | Qwen2-7B-Instruct, revision `f2826a0` | YAML `model_name_or_path: Qwen/Qwen2-7B-Instruct` |
| VSI-Bench revision | `d7cb1a3` | dataset revision the run loaded |
| Folds | k = 5 | cross-validation (`--n_splits 5`) |
| Seed | 42 | fold split (`--random_state 42`) and YAML `seed: 42` |
| Repeats | 1 | cross-validation (`--repeats 1`) |
| LoRA rank | r = 8 | YAML `lora_rank: 8` |
| LoRA alpha | α = 16 | YAML `lora_alpha: 16` |
| LoRA dropout | 0.1 | YAML `lora_dropout: 0.1` |
| LoRA targets | all modules | YAML `lora_target: all` |
| Epochs | one per fold | YAML `num_train_epochs: 1` |
| Train batch size | 4 | YAML `per_device_train_batch_size: 4` |
| Evaluation batch size | 4 | YAML `per_device_eval_batch_size: 4`; vLLM inference batches of 4 |
| Gradient accumulation | 1 | YAML `gradient_accumulation_steps: 1` |
| Learning rate | 2 × 10⁻⁴ | YAML `learning_rate: 0.0002` |
| Schedule | cosine decay | YAML `lr_scheduler_type: cosine` |
| Warmup | 0.1 | YAML `warmup_ratio: 0.1` |
| Precision | FP16 | YAML `fp16: true` |
| Maximum sequence length | 1024 | YAML `cutoff_len: 1024`; vLLM `max_model_len=1024` |
| Validation split | 10% of each training fold | YAML `val_size: 0.1` |
| Training template | `gemma` | YAML `template: gemma` |
| Inference prompt | raw prompt, no chat template | vLLM, `apply_chat_template=False` |
| Temperature | 0 | vLLM `temperature=0.0` |
| Top-p | 1 | vLLM `top_p=1.0` |
| New tokens | at most 10 | vLLM `max_tokens=10` (generated answers) |
| Log probabilities | top 100 | vLLM `logprobs=100` |
| MC scoring | probability of the gold option | see "Scoring" below |
| NUM scoring | generated answer, mean relative accuracy | see "Scoring" below |
| Held-out scoring | each sample is scored only in its held-out fold | cross-validation |
| GPUs | one | LoRA training runs on one visible GPU (see below) |

The command for the D.1 runs; every flag after `--mode llm` is a default:

```bash
uv run python -m TsT --benchmark vsi --mode llm --llm_model Qwen/Qwen2-7B-Instruct --n_splits 5 --repeats 1
```

## Training details

- **One adapter per fold.** Each fold's training split becomes a fresh LoRA adapter on
  the base model. A zero-shot pass of the base model gives the baseline.
- **Data format.** `<fold_tmp>/dataset/custom_sft_dataset.jsonl` holds one
  `{"instruction": <text-only prompt>, "output": <answer>}` record per training
  question. `dataset_info.json` registers it as:

  ```json
  {"custom_sft_dataset": {"file_name": "custom_sft_dataset.jsonl", "formatting": "alpaca",
    "ranking": false, "load_from": "file", "columns": {"prompt": "instruction", "response": "output"}}}
  ```

  The prompt is `Answer the following question: {question}`, followed for MC
  questions by " Options:" and one lettered option per line, then "Answer with the
  option's letter from the given choices directly." (MC) or "Answer with just a
  number." (NUM). The answer is the gold letter or number. See
  `src/TsT/evaluators/llm/data/conversion.py`.
- **Training template versus inference prompt (as run).** Training formats each
  example with LLaMA-Factory's `gemma` template, while inference sends the raw prompt
  without a chat template. This mismatch is how the paper's runs were configured, and
  App. D.1 states both; the defaults keep it so the paper's numbers reproduce.
- **One GPU.** With several visible GPUs, LLaMA-Factory would launch distributed
  training on all of them and multiply the effective batch size. The trainer
  therefore exposes only the first visible GPU to the training subprocess; vLLM also
  runs on one GPU. The paper's runs used one GPU.
- **Validation split.** `val_size: 0.1` makes LLaMA-Factory hold out 10% of each
  training fold (split with the run seed), and the adapter does not train on it. The
  config sets no `eval_strategy`, so training never evaluates on that split, and
  `eval_steps: 100` has no effect.
- **Checkpoints and logging.** `save_steps: 500`, `save_total_limit: 1`,
  `logging_steps: 10`, and 4 workers each for data loading and preprocessing. None of
  these change the result.

## Scoring

- **MC.** A one-token greedy pass requests the top-100 log probabilities at the first
  generated position. The score is the probability of the gold option letter,
  normalized over the option letters found among those 100 tokens. When the gold
  letter is not among them, the sample is scored 0 or 1 from the parsed answer, and an
  unparseable answer falls back to a random option. This release seeds that random
  choice by the run seed and the question, so it no longer varies between reruns. The paper's
  runs drew it unseeded, which moved zero-shot scores by up to a few tenths of a point between
  runs. On CV-Bench many zero-shot answers start with words rather than a letter, so the
  fallback gives them chance credit; App. D.1 reports the effect (+13.1 as scored, about
  +18.4 if such answers count as wrong).
- **NUM.** The model generates up to 10 tokens. The number parsed from the answer
  (number words are converted to digits) is scored with mean
  relative accuracy over thresholds 0.50, 0.55, …, 0.95, as implemented in this
  package. The package counts a threshold as met when the relative error is strictly
  below 1 − θ; official VSI-Bench uses ≤. An answer with no number scores 0 (in the
  paper's runs no answer lacked a number, so this does not change their scores).
- **Aggregation.** Each sample is scored once, in the fold where it is held out.
