# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the codebase for the NeurIPS 2025 position paper "Benchmark Designers Should 'Train on the Test Set' to Expose Exploitable Non-Visual Shortcuts". The project implements Test-Set Training (TsT) methodology to evaluate and mitigate statistical biases in vision-language benchmarks.

## Setup and Environment

The project uses Python 3.10+ with uv for package management. Setup requires special handling for flash-attn:

```bash
# Install dependencies 
uv python install 3.10
uv venv
uv pip install setuptools torch
uv sync
source .venv/bin/activate
```

Pre-download datasets using hf_transfer for faster downloads:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset nyu-visionx/VSI-Bench
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset nyu-visionx/CV-Bench
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset lmms-lab/Video-MME
```

## Development Commands

### Running Evaluations
```bash
# Main TsT evaluation script
uv run scripts/run_tst.py --benchmark {vsi|cvb|video_mme|mmmu} [options]

# Random Forest mode (default)
uv run scripts/run_tst.py --benchmark vsi --verbose
uv run scripts/run_tst.py --benchmark cvb --n_splits 10 --repeats 3
uv run scripts/run_tst.py --benchmark vsi --question_types "object_counting,object_abs_distance"

# LLM mode (new - uses LoRA fine-tuning)
uv run scripts/run_tst.py --benchmark vsi --mode llm --llm_model "google/gemma-2-2b-it"
uv run scripts/run_tst.py --benchmark mmmu --mode llm --llm_batch_size 2 --llm_epochs 1
uv run scripts/run_tst.py --benchmark video_mme --mode llm --verbose
```

### Code Quality
```bash
# Linting (configured in pyproject.toml with line-length 120, excludes notebooks)
ruff check
ruff format

# Testing
pytest tests/
```

### Running Legacy Debiasing
```bash
# Generate removed IDs for VSI debiasing (legacy functionality)
python debias_vsi_clean.py
```

## Architecture

### Core Components

**`src/TsT/`** - Main package containing the TsT evaluation framework:
- `evaluation.py`: Core evaluation functions (`run_evaluation`, `evaluate_bias_model`, `evaluate_bias_model_llm`)
- `llm_utils.py`: LLM integration utilities (LLaMA-Factory wrapper, dataset conversion, LoRA training)
- `protocols.py`: QType protocol defining the interface for question types
- `utils.py`: Utility functions

**`src/TsT/benchmarks/`** - Benchmark-specific implementations:
- `vsi/`: VSI-Bench (Visual Spatial Intelligence) with 8 question types
- `cvb/`: CV-Bench (Computer Vision Benchmark) with 4 question types  
- `video_mme/`: Video-MME benchmark
- `mmmu/`: MMMU benchmark

Each benchmark contains:
- `data_loader.py`: Dataset loading and preprocessing
- `models.py`: Question type implementations (QType protocol)

**`scripts/run_tst.py`** - Unified evaluation runner that dynamically imports benchmark modules

### Question Type Protocol

All question types implement the `QType` protocol:
- `name`: Question type identifier
- `format`: "mc" (multiple choice) or "num" (numerical)
- `feature_cols`: List of feature column names
- `select_rows()`: Filter dataset for this question type
- `fit_feature_maps()`: Learn statistics from training data only
- `add_features()`: Add question-specific features to dataframe

### TsT Methodology

The core TsT evaluation supports two modes:

**Random Forest Mode (default):**
1. Uses Random Forest models (classification for MC, regression for numerical)
2. Cross-validation with stratified splits for MC questions
3. Metrics: accuracy for MC, mean relative accuracy for numerical
4. Feature engineering based on statistical patterns in questions

**LLM Mode (new):**
1. Uses LoRA fine-tuning on small LLMs (e.g., Gemma-2-2B) for bias detection
2. Converts datasets to "blind QA" format (text-only, no visual information)
3. Trains separate LoRA adapters for each k-fold split
4. Reports improvement over zero-shot baseline to isolate bias-specific learning
5. Efficient training with QLoRA, Flash Attention, and minimal epochs (typically 1)

## Data Structure

- `data/`: Contains evaluation datasets and reference model results
- `data/ref_evals/`: Pre-computed model evaluations for comparison
- `notebooks/`: Jupyter notebooks for analysis and development
- `logs/`: Evaluation logs organized by benchmark

## External Dependencies

- **LLaMA-Factory**: Located at `src/external/LLaMA-Factory/` as an editable submodule for fine-tuning capabilities
- Uses UV's build isolation exclusion for flash-attn compilation requirements