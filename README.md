# TsT: Test-Set-Training

Code for the paper `Benchmark Designers Should ``Train on the Test Set'' to Expose Exploitable Non-Visual Shortcuts`.



## Setup

```bash
# 1. Clone the repository
git clone git@github.com:vision-x-nyu/test-set-training.git
cd test-set-training

# 2. Install dependencies in a venv
uv python install 3.9  # ensure using uv's cpython version over default distro
uv sync

# 3. Activate the environment
source .venv/bin/activate
```

### Pre-Download Datasets (recommended)
Using `hf_transfer` can speed up dataset downloads dramatically versus the default single-stream downloader used by the python `datsets` library.

```bash
# VSI-Bench
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset nyu-visionx/VSI-Bench

# CV-Bench
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset nyu-visionx/CV-Bench

# Video-MME
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset lmms-lab/Video-MME
```

## TsT

### Running Evaluations

Use the unified script to run evaluations on any benchmark:

```bash
# VSI-Bench
uv run scripts/run_tst.py --benchmark vsi

# CV-Bench  
uv run scripts/run_tst.py --benchmark cvb

# Video-MME
uv run scripts/run_tst.py --benchmark video_mme
```

#### Additional Options

```bash
# With custom parameters
uv run scripts/run_tst.py --benchmark video_mme --n_splits 10 --verbose --repeats 3

# Evaluate specific question types only
uv run scripts/run_tst.py --benchmark vsi --question_types "object_counting,object_size_estimation"
```

## IBP
TODO

<br><br><br>
<br><br><br>

# [OLD] ~~VSI-Bench Debiased (Internal)~~

> [!WARNING]  
> The below details are out of date.
>
> **TODO:** update with IBP code / instructions

This repository contains code and analysis for creating a debiased version of the VSI-Bench dataset. The goal is to mitigate statistical biases and shortcuts present in the original benchmark, ensuring that models are evaluated more accurately on their visual-spatial reasoning capabilities rather than their ability to exploit non-visual patterns.

## Background

Initial experiments with models trained on simulated data for VSI-Bench (SIMS) and real in-distribution VSI training data (VSI-Train) reveal the presence of significant exploitable statistical biases within the VSI-Bench dataset itself (e.g., imbalanced answer distributions, correlations between object types and expected values/answers).

This framework implements various strategies to identify and filter out potentially biased samples from the original VSI-Bench dataset, aiming to produce a more robust v1.1 evaluation set.

## Approach

The core approach involves analyzing each question type within VSI-Bench and applying a tailored filtering strategy:

1.  **Bias Identification:** For each question type, potential statistical shortcuts were identified (e.g., frequency of specific answers, typicality of numerical values, predictability based on object types).
2.  **Scoring/Ranking:** Functions were developed (`get_*_score`) to assign a bias score or calculate metrics for each question based on these potential biases. These scores often leverage frequencies, distances from means (sometimes in log-space), or relative comparisons between ground truth and distractor options. For some tasks, machine learning models (like Random Forests) were used diagnostically to identify highly predictable features.
3.  **Filtering:** Filtering functions (`filter_*`) use the calculated scores or metrics to select samples for removal. Different strategies are employed based on the bias type:
    * **Score-based Truncation/Sampling:** Removing samples with the highest bias scores (e.g., `object_size_estimation`, `object_abs_distance`, `obj_appearance_order`). Weighted sampling is used for `room_size_estimation` to avoid creating unnatural distributions.
    * **Dynamic Balancing:** Iteratively removing samples from over-represented groups (e.g., specific GT answers in `object_rel_direction`, `route_planning`, or GT values in `object_counting`) until a target distribution or threshold is met, using a secondary bias score to prioritize *which* samples within the group to remove.
4.  **Evaluation Integration:** The framework includes utilities to load reference model evaluations and compare performance metrics before and after debiasing to guide the process and assess the impact of filtering.

## Repository Structure

* [`debias_vsi_clean.py`](debias_vsi_clean.py): The main, cleaned-up Python script containing the final scoring and filtering functions for each question type, along with the main experiment execution logic.
* [`debias_vsi.py`](debias_vsi.py): The original development script containing numerous experimental versions of functions (kept for reference).
* [`data/`](data/):
    * [`ref_evals/`](data/ref_evals/): Contains JSON(L) files with pre-computed scores for various reference models on the original VSI-Bench. Used for evaluating the impact of debiasing.
    * [`removed_ids.txt`](data/removed_ids.txt): The output file generated by `debias_vsi_clean.py`, listing the IDs of the questions removed from the original dataset.
* [`notebooks/`](notebooks/): Contains Jupyter notebooks used for exploratory data analysis, development, and debugging of the filtering logic for each question type.
    * [`vsi_debias__combined.ipynb`](notebooks/vsi_debias__combined.ipynb): Runs the end-to-end debiasing process interactively.
    * Individual notebooks (`vsi_debias_obj_count.ipynb`, etc.) focus on specific question types.



## Running the Debiasing Script

The primary script to run the full debiasing process with the finalized parameters is [`debias_vsi_clean.py`](debias_vsi_clean.py). The final list of removed question IDs  will be saved to [`data/removed_ids.txt`](data/removed_ids.txt).

```bash
python debias_vsi_clean.py
```

> **Note:** The script uses hardcoded parameters determined after extensive experimentation (see the notebooks for details).


## Output

The primary output is [`data/removed_ids.txt`](data/removed_ids.txt). This file contains the unique IDs of all questions identified for removal by the script


## Filtering Strategies Summary

The final filtering strategies implemented in `debias_vsi_clean.py` are:


| **Question Type**         | **Filtering Strategy**                                                                                                                                                                                                 |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Object Size Estimation** | Score-based removal prioritizing samples close to the object's mean size, considering object frequency and size variance.                                                                                            |
| **Object Counting**        | Dynamic balancing to cap GT value frequency below a threshold (e.g., 20%), prioritizing removal based on object/combo frequency score.                                                                               |
| **Object Appearance Order**| Score-based removal using a relative score comparing the GT sequence's statistical likelihood (positional + pair frequency) to the distractors'.                                                                    |
| **Object Absolute Distance**| Score-based removal using object pair frequency, pair distance variance, and closeness to the pair's mean distance (in log-space), plus closeness to the global mean distance (in log-space).                        |
| **Object Relative Distance**| Score-based removal using a combination of max option probabilities derived from GT object frequency, sorted pair frequency, and ordered pair frequency.                                                             |
| **Object Relative Direction**| Answer distribution balancing (per subtype: easy/medium/hard) by removing excess samples from frequent answers, prioritizing removal based on the frequency of objects involved in the question.                     |
| **Room Size Estimation**   | Weighted sampling based on closeness to the global mean size (in log-space), using the log-normal PDF fit to concentrate removal near the mean.                                                                      |
| **Route Planning**         | Answer letter ('A'/'B'/'C'/'D') distribution balancing, prioritizing removal based on a secondary score derived from object frequency, route string frequency, and step count typicality.                             |

## Debiased Dataset Statistics
The table below summarizes the number of samples in the original dataset, the number of samples filtered out, and the number of samples remaining after debiasing for each question type.

| **Question**                         | **Fmt** | **# Orig** | **% Orig** | **# Filter** | **# Deb** | **% Deb** |
|--------------------------------------|--------:|---------:|-----------:|-----------:|--------:|----------:|
| object_size_estimation               |     oe  |      953 |    18.58%  |        600 |     353 |    14.94% |
| object_abs_distance                  |     oe  |      834 |    16.26%  |        400 |     434 |    18.37% |
| object_rel_distance                  |     mc  |      710 |    13.84%  |        400 |     310 |    13.12% |
| obj_appearance_order                 |     mc  |      618 |    12.05%  |        300 |     318 |    13.46% |
| object_counting                      |     oe  |      565 |    11.01%  |        314 |     251 |    10.63% |
| object_rel_direction_medium          |     mc  |      378 |     7.37%  |        324 |      54 |     2.29% |
| object_rel_direction_hard            |     mc  |      373 |     7.27%  |        257 |     116 |     4.91% |
| object_rel_direction_easy            |     mc  |      217 |     4.23%  |          5 |     212 |     8.98% |
| room_size_estimation                 |     oe  |      288 |     5.61%  |         88 |     200 |     8.47% |
| route_planning                       |     mc  |      194 |     3.78%  |         80 |     114 |     4.83% |
| **Total**                            |         | **5130** | **100.00%** | **2768** | **2362** | **100.00%** |
