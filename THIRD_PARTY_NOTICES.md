# Third-party notices

Copyright 2025-2026 The TsT Authors. This repository is licensed under Apache-2.0 (see
[LICENSE](LICENSE)). It adapts code from the projects below and includes per-question data from
the datasets listed under [Data](#data); datasets, model weights and third-party code remain
subject to their own terms.

## DataEnvGym (MIT License)

`src/TsT/evaluators/llm/trainers/llamafactory.py` adapts the LLaMA-Factory dataset preparation
and training call from DataEnvGym (<https://github.com/codezakh/dataenvgym>, commit
`f698f39c7d77fc655942099535d06a4d11b32e3b`, `llama_factory_utils.py`), and
`src/TsT/evaluators/llm/predictors/vllm.py` follows its vLLM LoRA predictor
(`trainable_predictors/common.py`).

```
MIT License

Copyright (c) 2024 Zaid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MMMU evaluation code (Apache-2.0)

The multiple-choice answer parser in `src/TsT/evaluators/llm/scoring.py` is adapted from the
MMMU evaluation code (<https://github.com/MMMU-Benchmark/MMMU>, commit
`51ce7f3e829c16bb44bc5445782686b4c3508794`, `eval/data_utils.py` and `eval/eval_utils.py`).

## Thinking in Space (Apache-2.0)

The answer-matching and mean-relative-accuracy helpers in `src/TsT/utils.py`,
`reproduce/robust_num.py` and `legacy/debias_vsi_clean.py` are adapted from the VSI-Bench
evaluation code in <https://github.com/vision-x-nyu/thinking-in-space>.

## Data

`reproduce/data/` and `tests/fixtures/` hold per-question data from these datasets (no images or
videos):

- **VSI-Bench** and **VSI-Bench-Debiased** (<https://huggingface.co/datasets/nyu-visionx/VSI-Bench>)
  and **CV-Bench** (<https://huggingface.co/datasets/nyu-visionx/CV-Bench>): question ids, gold
  answers and model predictions (`reproduce/data/preds/`, `reproduce/data/gpt4o/vsi_mc.jsonl.gz`,
  `reproduce/data/gpt4o/cvb_mc.jsonl.gz`), question-id lists (`reproduce/data/ibp/`,
  `reproduce/data/vsi_bench_debiased_v1_removed_ids.txt`), and small question-text subsets in
  `tests/fixtures/`. See the dataset cards for their terms; the VSI-Bench videos remain subject to
  the ScanNet, ScanNet++ and ARKitScenes terms of use.
- **MMMU** (Yue et al., 2024; <https://huggingface.co/datasets/MMMU/MMMU>, Apache-2.0; read from the
  `lmms-lab/MMMU` copy on Hugging Face): validation question ids with gold answer letters and
  GPT-4o's answers (`reproduce/data/gpt4o/mmmu_mc.jsonl.gz`), per-question correctness of
  LLaVA-OneVision-7B (`reproduce/data/ibp/mmmu/`), and a list of question ids
  (`reproduce/data/ibp/llm/mmmu_delta_removed_ids.txt`).
- **Video-MME**: question ids and GPT-4o's answers only, in
  `reproduce/data/gpt4o/video_mme_mc.jsonl.gz`. No Video-MME questions, options or gold answers
  are included; the scripts download them from Hugging Face when needed.
- **MMStar**: no MMStar data is included.
