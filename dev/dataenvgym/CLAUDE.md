# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
# Install dependencies with uv
uv sync requirements.txt

# If type-checking issues arise with VSCode, use:
pip install -e . --config-settings editable_mode=compat --force-reinstall
```

### Testing
```bash
# Run all tests
pytest

# Run specific test directory
pytest tests/unit/
pytest tests/integration/

# The pytest configuration excludes the 'commands' directory from test discovery
```

### Running Examples
```bash
# Run examples from repository root with GPU specification
CUDA_VISIBLE_DEVICES=<gpu_ids> python examples/math/open_ended_environment.py
CUDA_VISIBLE_DEVICES=<gpu_ids> python examples/gqa/skill_tree_environment.py

# Make sure to set num_gpus in ray.init() to match available GPUs
```

### LiveCodeBench Evaluation
```bash
# Run LiveCodeBench evaluation command
python commands/run_livecodebench_evaluation.py
```

## Architecture

DataEnvGym is a framework for creating and evaluating data generation agents that improve student models through iterative training. The core abstractions are:

### Core Interfaces

1. **TaskInterface** (`dataenvgym/gym/tasks/`): Represents collections of task instances (VQA questions, MATH problems, etc.)
2. **TrainablePredictorInterface** (`dataenvgym/gym/trainable_predictors/`): Models that can be trained on sequences of training data
3. **DataGenerationAgentInterface** (`dataenvgym/gym/data_generation_agents/`): Agents that generate new training data
4. **EnvironmentInterface** (`dataenvgym/gym/environments/`): Environments for evaluating data generation agents

### Key Components

- **Tasks**: MATH, GQA, LiveCodeBench, MnMs (tool use), NaturalBench
- **Trainable Predictors**: Support for local LLMs, vLLM, BLIP, PaliGemma
- **Data Generation Agents**: Baselines for open-ended, skill-list, and skill-tree approaches
- **Integration**: Built-in support for vLLM, Ray, and LLaMA-Factory

### Directory Structure

- `src/dataenvgym/gym/`: Core gym components
- `examples/`: Working examples for each task type
- `tests/`: Unit and integration tests
- `src/external/LLaMA-Factory/`: Integrated LLaMA-Factory for training
- `commands/`: Standalone command scripts

## Environment Setup

### Required Environment Variables
- For Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`  
- For OpenAI: `OPENAI_API_KEY`

### Ray and GPU Configuration
The framework uses Ray for distributed computing. Always set `num_gpus` in `ray.init()` to match your available GPUs and use `CUDA_VISIBLE_DEVICES` to specify which GPUs to use.

## Development Notes

- The codebase is fully typed
- HuggingFace Datasets automatically downloads required datasets
- MATH dataset has known issues with HuggingFace availability
- Examples should be run from the repository root directory
- The base environment (`base_environment.py`) supports all environment types (skill-list, skill-tree, open-ended) through configuration