<div align="center">


<!-- TITLE -->
# TsT: Test-Set Stress-Test

<!-- BADGES -->

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv:2511.04655-b31b1b.svg?style&logo=arXiv)](https://arxiv.org/abs/2511.04655)
[![arXiv](https://img.shields.io/badge/📄_PDF-TsT-FDDEB3.svg)](https://arxiv.org/pdf/2511.04655)
[![Project](https://img.shields.io/badge/🌎_Web-Test--Set_Stress--Test-blue.svg)](https://vision-x-nyu.github.io/test-set-training/)
[![Home](https://img.shields.io/badge/HF-VSI--Bench_(Debiased)-FED123.svg?style&logo=HuggingFace)](https://hf.co/datasets/nyu-visionx/VSI-Bench)

</div>


<!-- DESCRIPTION -->
## Abstract
Robust benchmarks are crucial for evaluating Multimodal Large Language Models (MLLMs). Yet we find that models can ace many multimodal benchmarks without strong visual understanding, instead exploiting biases, linguistic priors, and superficial patterns. This is especially problematic for vision-centric benchmarks that are meant to require visual inputs. We adopt a diagnostic principle for benchmark design: if a benchmark can be gamed, it will be. Designers should therefore try to “game” their own benchmarks first, using diagnostic and debiasing procedures to systematically identify and mitigate non-visual biases. Effective diagnosis requires directly “training on the test set”—probing the released test set for its intrinsic, exploitable patterns.

We operationalize this standard with two components. First, we diagnose benchmark susceptibility using a “Test-set Stress-Test” (TsT) methodology. Our primary diagnostic tool involves fine-tuning a powerful Large Language Model via k-fold cross-validation on exclusively the non-visual, textual inputs of the test set to reveal shortcut performance and assign each sample a bias score s(x). We complement this with a lightweight Random Forest-based diagnostic operating on hand-crafted features for fast, interpretable auditing. Second, we debias benchmarks by filtering high-bias samples using an “Iterative Bias Pruning” (IBP) procedure. Applying this framework to four benchmarks—VSI-Bench, CV-Bench, MMMU, and VideoMME—we uncover pervasive non-visual biases. As a case study, we apply our full framework to create VSI-Bench-Debiased, demonstrating reduced non-visual solvability and a wider vision-blind performance gap than the original.


## Code
Coming soon!


<!-- CITATION -->
## Citation

```bibtex
@article{brown2025benchmark,
  author = {Brown, Ellis and Yang, Jihan and Yang, Shusheng and Fergus, Rob and Xie, Saining},
  title = {Benchmark Designers Should ``Train on the Test Set'' to Expose Exploitable Non-Visual Shortcuts},
  journal = {arXiv preprint arXiv:2511.04655},
  year = {2025},
}
```

