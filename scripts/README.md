# Test-Set Training Scripts

This directory contains one-off scripts, examples, and experiments.

> **Note:** The recommended way to run TsT evaluations is now via the package CLI: `uv run python -m TsT`

## Usage

### General Benchmark Runner

```bash
# Run VSI benchmark
uv run python -m TsT --benchmark vsi --verbose

# Run CVB benchmark  
uv run python -m TsT --benchmark cvb --verbose

# Run specific question types
uv run python -m TsT --benchmark vsi --question_types object_counting,object_abs_distance

# Run with custom parameters
uv run python -m TsT --benchmark cvb --n_splits 10 --repeats 3 --random_state 123
```

### Arguments

- `--benchmark, -b`: The benchmark to run (`vsi` or `cvb`) - **required**
- `--n_splits, -k`: Number of cross-validation splits (default: 5)
- `--random_state, -s`: Random seed (default: 42)
- `--verbose, -v`: Print detailed output
- `--repeats, -r`: Number of evaluation repeats (default: 1)
- `--question_types, -q`: Comma-separated list of specific question types to evaluate
- `--target_col, -t`: Target column name (defaults to benchmark-specific default)

### Available Benchmarks

#### VSI (Visual Spatial Intelligence)
- **Question Types**: object_counting, object_abs_distance, object_size_estimation, room_size_estimation, object_rel_distance, object_rel_direction, route_planning, obj_appearance_order
- **Default Target**: `ground_truth` (for numerical questions), `gt_idx` (for multiple choice questions)

#### CVB (Computer Vision Benchmark)  
- **Question Types**: count_2d, relation_2d, depth_3d, distance_3d
- **Default Target**: `gt_idx`

### Examples

```bash
# Quick test run
uv run python -m TsT --benchmark vsi --question_types object_counting --n_splits 2

# Full evaluation with multiple repeats
uv run python -m TsT --benchmark cvb --verbose --repeats 5 --n_splits 10

# Run only spatial reasoning questions
uv run python -m TsT --benchmark vsi --question_types object_rel_distance,object_rel_direction,route_planning
``` 