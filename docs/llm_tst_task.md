# Goal: Implement LLM-Finetuning

the current paper draft is here: ./draft.tex

it was a quick rush to pull this paper together by the neurips deadline and we now need to improve it now before arxiving.

prob the biggest flaw is that we only validate the TsT method on two benchmarks: VSI-Bench and CV-Bench. these benchmarks
* were produced by US (my own group at NYU)
* are "template-based" --- ie, they were programmatically generated using set templates for different question types.

so one primary area of focus for improvement right now is getting results on more datasets that we didnt create and that also have less well-defined, exploitable structure in the question text.

because I first focused on this using the template datasets, I landed upon the random forest methodology (vaguely) described in the paper. this is nice because they are very fast to fit---a requirement of my TsT methodology, wherein you have to FT your model *k* times, on each of the k-fold CV splits. it also gives very interpretable insights, with feature importances and has a well-defined "confidence" measure which I can use downstream in the Iterative Bias Pruning method (not currently implemented here)
the "TsT method with random forest is implemented in this file currently: [src/TsT/evaluation.py](../src/TsT/evaluation.py)

I have unfortunately been struggling to get as strong of results with this random forest approach on the [Video-MME](../src/TsT/benchmarks/video_mme/) and [MMMU](../src/TsT/benchmarks/mmmu/) benchmarks. this is in large part because it is nearly impossible programmatically extract features from the questions and answer options when the QAs were written by a human or llm rather than generated via a template, so we can only rely on the external metadata. this *can* draw out interesting biases, but does not learn anything about the *specific* sample.

we have been discussing trying to slot in an LLM in place of the old school decision tree in our method. in fact, saining raises a great point about this.

> → if your benchmark is for LLMs, you should be trainingin an LLM (not a Random Forest). This should be the default setting.

I agree, if we can get good results. we can still include the RF setting as a way to get additional / more interpretable insights.

and we are testing VQA benchmarks designed to test mutlimodal LLMs. this approach might find much wider usage in the community if we show results finetuning LLMs in the method.

(minor aside: one minor concern I had previously is that the LLMs themselves have strong language priors and decent baseline blind performance already. to mitigate this, I think we can focus on the *improvement* over the blind zero-shot baseline achieved by the BLIND LLM finetuning (or lora tuning) with the cross-validation procedure.)

if the core premise of our method holds---that NON-VISUAL shortcuts can be identified by doing this cross-validation training DIRECTLY on the test set---then this should work too.

this is where the efficiency comes in. depending on how many folds we want to do (say, k=5 by default?), we will have to finetune and inference *k* separate LLMs.
every part of this needs to be efficient:
* training the model on the k-1 folds --> this needs to be especially efficient
* inferencing the model on the 1 left out fold --> this can be slightly slower
* saving the model down. we cant afford to save down 1000s of GB of model params when running this!
so ideally we can use all of the training time speed up tricks (lora/qlora/flash-attn/etc).

I've looked into a bunch of different LLM training frameworks and chatted with some friends. one good friend *STRONGLY* recommended using `llamafactory`.
he pointed me to his code repo where he heavily used it with a somewhat similar purpose. he was also generating data on-the-fly, and using llamafactory to train models on the data.
I've added his codebase to our project dir for our internal reference here: [src/reference/dataenvgym](../src/reference/dataenvgym)
I have no plans for us to run it.

I have also had claude code examine the repo and produce some notes:
* general notes: [src/reference/dataenvgym/CLAUDE.md](../src/reference/dataenvgym/CLAUDE.md)
* notes on the repo's usage of llamafactory + on-the-fly datasets: [src/reference/dataenvgym/2025-07-24-command-messageinit-is-analyzing-your-codebase.txt](../src/reference/dataenvgym/2025-07-24-command-messageinit-is-analyzing-your-codebase.txt)

I originally asked ChatGPT o3 about what we should do and it proposed some solid ideas (transcript below). however, I think we should reference our implementation more closely on the dataenvgym implementation.



---

# ChatGPT o3 thoughts:

Below is a practical roadmap for swapping your Random-Forest “blind” predictor with a **tiny LoRA-tuned LLM** inside the TsT pipeline, while keeping the whole k-fold loop light enough to run on a single workstation or a pair of rented GPUs.

---

## 1  Pick a “good-enough” base model

| Size          | Good defaults                    | VRAM (QLoRA, 4-bit) | LoRA-adapter size (r = 8) | Fold-train wall-time\* |
| ------------- | -------------------------------- | ------------------- | ------------------------- | ---------------------- |
| **2 B-class** | `google/gemma-2b-it` or `phi-2`  | 6-7 GB              | ≈ 8 MB                    | 2–3 min                |
| **7 B-class** | `mistralai/Mistral-7B-v0.2`      | 14-16 GB            | ≈ 28 MB                   | 7–10 min               |
| **8 B-class** | `meta-llama/Llama-3-8b-instruct` | 18 GB               | ≈ 32 MB                   | 9–12 min               |

\*Measured on one RTX 4090 using LLaMA-Factory with Flash-Attention 2 + Unsloth.

**Why not larger?**
Your TsT loop scales training cost × *k*. Five folds with a 7 B base already give 50–60 min total run-time; bigger models buy little in bias detection but explode compute.

---

## 2  Convert the benchmark to a “blind-QA” fine-tuning set

For each sample keep **only**:

```json
{
  "instruction": "<question text>",
  "input": "",                     # leave empty – no image/video tokens
  "output":   "<ground-truth answer>"
}
```

* Stratify the k-fold split by answer label to avoid majority-class leakage.
* For multiple-choice (Video-MME, MMMU) concatenate the options so the model must pick the correct label verbatim:

```
Question: …? Choices: (A) cat (B) dog (C) car
```

---

## 3  One-command fold trainer (LLaMA-Factory)

Create a shell script that loops over folds:

```bash
#!/usr/bin/env bash
MODEL=google/gemma-2b-it
for FOLD in {0..4}; do
  python src/train.py \
    --model_name $MODEL \
    --dataset_dir ./fold_$FOLD/train \
    --val_dataset_dir ./fold_$FOLD/val \
    --finetuning_type lora \
    --lora_r 8 --lora_alpha 16 \
    --quantization_bit 4 \
    --batch_size 32 --epochs 1 \
    --learning_rate 2e-4 \
    --max_seq_length 512 \
    --output_dir ./adapters/fold_$FOLD
done
```

*`epochs = 1` is often enough—remember each fold still sees 80 % of the test set.*

### Speed boosts

* **Unsloth backend** (`--use_unsloth`) → \~1.4× faster compilation-free training.
* **GaLore or DoRA** flags if you want even tinier adapters, but r = 8 already < 10 MB.
* Set `--flash_attn true` (enabled by default on FA2-capable GPUs).

---

## 4  Inference & scoring

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
bnb = BitsAndBytesConfig(load_in_4bit=True)
base = AutoModelForCausalLM.from_pretrained(MODEL,
                                            quantization_config=bnb,
                                            device_map="auto")

for fold in range(5):
    adapter = PeftModel.from_pretrained(base, f"./adapters/fold_{fold}")
    merged  = adapter.merge_and_unload()      # keeps inference speed native
    # batch-generate on the held-out fold, pick first token after BOS, compare to GT …
```

*Save only the `.safetensors` LoRA weights*—five folds of Gemma-2B is < 50 MB total.

---

## 5  Metrics to report in the paper

| What                                                | Why it matters                                         |
| --------------------------------------------------- | ------------------------------------------------------ |
| **Blind zero-shot baseline** (no fine-tune)         | quantifies innate language priors                      |
| **TsT-LoRA accuracy (µ ± σ over folds)**            | your new non-visual solvability number                 |
| **∆ = TsT-LoRA – zero-shot**                        | isolates the gain from test-set-specific bias          |
| **Vision / Blind gap on debiased vs. original set** | shows downstream impact exactly as in VSI-Bench-Robust |

Reviewers will accept the stronger blind prior—as long as you focus on *improvement* rather than raw score, exactly as you proposed.

---

## 6  Adapting Iterative Bias Pruning (IBP)

1. Use the model’s **predicted probability of the chosen answer** as *s(x)*. (With HF you get log-probs cheaply in 4-bit.)
2. Keep the same percentile-based removal schedule you described.
3. Re-train LoRA on the reduced set each iteration (still cheap).
4. Stop when TsT-LoRA drops to within, say, **+5 pp of chance** or when removal budget hits 20 %.

---

## 7  Interpretability hooks (optional but nice)

* Pull **feature attribution** from the final linear layer w\.r.t. the prompt tokens (`model.derive_gradients()` in Captum).
* Or fit a simple logistic regression on top of the LoRA-tuned **CLS embedding** to approximate token importance—gives you a heat-map to put in the appendix.

---

## 8  Sections to tweak in the draft

| Draft section                | Minimal edits                                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Sec 3.1 Diagnostic Model** | replace Random-Forest paragraph with: *“We instead fine-tune a 2 B-class LLM with LoRA (r = 8, 4-bit) on non-visual text via k-fold TsT…”* |
| **Algorithm 1**              | the `ComputeSampleBiasScores` call now refers to *LoRA-Tuned LLM*.                                                                         |
| **Experiments (new)**        | add Video-MME & MMMU table with *zero-shot*, *TsT-LoRA*, *Vision* rows.                                                                    |
| **Compute budget footnote**  | cite 5 folds × 3 min = 15 min on one 4090 (< \$0.60 on Lambda).                                                                            |
| **Discussion**               | note LoRA keeps adapters tiny (< 50 MB) so TsT scales to k = 10 without disk bloat.                                                        |

(Everything else—including IBP and debiasing analysis—remains unchanged.)&#x20;

---

### TL;DR

* **LoRA + QLoRA** lets you slot an LLM into TsT with < 10 MB per fold and minutes of training.
* Use a 2 B or 7 B base, Unsloth + Flash-Attention 2 for speed.
* Report *improvement over zero-shot* to neutralize inherent language priors.
* The rest of your framework (bias scores → IBP → vision/blind gap) transfers verbatim to new benchmarks.

Let me know if you’d like example scripts, ablation ideas, or help integrating these numbers into the paper!
