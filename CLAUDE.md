# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

### Before starting work
- Always in plan mode to make a plan
- After get the plan, make sure you Write the plan to .claude/tasks/TASK_NAME.md.
- The plan should be a detailed implementation plan and the reasoning behind them, as well as tasks broken down.
- If the task require external knowledge or certain package, also research to get latest knowledge (Use Task tool for research)
- Don't over plan it, always think MVP.
- Once you write the plan, firstly ask me to review it. Do not continue until I approve the plan.

### While implementing
- You should update the plan as you work.
- After you complete tasks in the plan, you should update and append detailed descriptions of the changes you made, so following tasks can be easily hand over to other engineers.


## Project Overview

This is the codebase for the NeurIPS 2025 position paper "Benchmark Designers Should 'Train on the Test Set' to Expose Exploitable Non-Visual Shortcuts". The project implements Test-Set Training (TsT) methodology to evaluate and mitigate statistical biases in vision-language benchmarks.

## Setup and Environment

The project uses Python 3.10 with uv for package management. **Important**: Use the bootstrap script for proper flash-attn + torch installation:

```bash
# Clone with submodules (required for LLaMA-Factory)
git clone --recurse-submodules <repo-url>
cd test-set-training

# Run bootstrap script (handles CUDA detection and torch installation)
bash scripts/bootstrap.sh

# Activate environment
source .venv/bin/activate
```

The bootstrap script automatically:
- Installs Python 3.10 via uv
- Detects CUDA version and installs appropriate torch wheel
- Handles no-build-isolation for flash-attn compilation

Pre-download datasets using hf_transfer for faster downloads:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset nyu-visionx/VSI-Bench
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset nyu-visionx/CV-Bench
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type=dataset lmms-lab/Video-MME
```

## Development Commands

### Running Evaluations
```bash
# Main TsT evaluation CLI
uv run python -m TsT --benchmark {vsi|cvb|video_mme|mmmu} [options]

# Random Forest mode (default)
uv run python -m TsT --benchmark vsi --verbose
uv run python -m TsT --benchmark cvb --n_splits 10 --repeats 3
uv run python -m TsT --benchmark vsi --question_types "object_counting,object_abs_distance"

# LLM mode (new - uses LoRA fine-tuning)
uv run python -m TsT --benchmark vsi --mode llm --llm_model "google/gemma-2-2b-it"
uv run python -m TsT --benchmark mmmu --mode llm --llm_batch_size 2 --llm_epochs 1
uv run python -m TsT --benchmark video_mme --mode llm --verbose
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
- `protocols.py`: QType protocol defining the interface for question types
- `utils.py`: Utility functions
- `llm/`: Modern distributed LLM integration (Ray + vLLM + LLaMA-Factory)
  - `interfaces.py`: Protocol-based interfaces for extensibility
  - `predictors.py`: Ray-based distributed inference with vLLM
  - `llamafactory_utils.py`: Clean LLaMA-Factory integration with Hydra configs
  - `data_conversion.py`: Blind QA format conversion utilities
  - `configs/`: Hydra configuration templates for different models

**`src/TsT/benchmarks/`** - Benchmark-specific implementations:
- `vsi/`: VSI-Bench (Visual Spatial Intelligence) with 8 question types
- `cvb/`: CV-Bench (Computer Vision Benchmark) with 4 question types  
- `video_mme/`: Video-MME benchmark
- `mmmu/`: MMMU benchmark

Each benchmark contains:
- `data_loader.py`: Dataset loading and preprocessing
- `models.py`: Question type implementations (QType protocol)

**`src/TsT/__main__.py`** - Unified evaluation CLI (`python -m TsT`) that dynamically imports benchmark modules
**`scripts/`** - One-off scripts, examples, and experiments

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

**LLM Mode (new - distributed architecture):**
1. Uses Ray + vLLM for 2-4x faster distributed inference across multiple GPUs
2. LoRA fine-tuning on small LLMs (e.g., Gemma-2-2B) with Hydra configuration management  
3. Converts datasets to "blind QA" format (text-only, no visual information)
4. Trains separate LoRA adapters for each k-fold split using LLaMA-Factory
5. Reports improvement over zero-shot baseline to isolate bias-specific learning
6. Automatic GPU memory management and worker cleanup between train/inference phases
7. Efficient training with QLoRA, Flash Attention 2, and minimal epochs (typically 1)

## Data Structure

- `data/`: Contains evaluation datasets and reference model results
- `data/ref_evals/`: Pre-computed model evaluations for comparison
- `notebooks/`: Jupyter notebooks for analysis and development
- `logs/`: Evaluation logs organized by benchmark

## External Dependencies

- **LLaMA-Factory**: Located at `src/external/LLaMA-Factory/` as an editable submodule for fine-tuning capabilities
- Uses UV's build isolation exclusion for flash-attn compilation requirements
- **CUDA Support**: Bootstrap script auto-detects CUDA version (11.8, 12.1, 12.2, 12.4, 12.8) and installs matching torch wheels

## Important Environment Notes

- **Always use `uv` to run commands** - ensures proper environment activation and dependency resolution
- **Package management**: Use `uv add <package>` to add new dependencies
- **Flash Attention**: Requires no-build-isolation handling (automated by bootstrap script)
- **Ray/vLLM**: Distributed inference requires adequate GPU memory across workers