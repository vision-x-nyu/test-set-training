#!/bin/bash

# 0. ensure using uv's cpython version over default distro
uv python install 3.10

# 1. create / activate env ----------------------------------------------------
uv venv
source .venv/bin/activate

# 2. detect CUDA runtime + pick a torch wheel
CUDA_VERSION=$(nvidia-smi --version | grep 'CUDA Version' | awk -F': ' '{print $2}')

case $CUDA_VERSION in
  12.8) TORCH="torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128" ;;
  12.4) TORCH="torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124" ;;
  12.1|12.2) TORCH="torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121" ;;
  11.8) TORCH="torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118" ;;
  *) echo "⚠️  Unrecognised CUDA $CUDA_VERSION → falling back to source build"; TORCH="" ;;
esac

echo "Detected CUDA version: $CUDA_VERSION"
echo "Installing torch wheel: $TORCH"

uv pip install setuptools packaging ninja         # build helpers
[ -n "$TORCH" ] && uv pip install $TORCH          # GPU wheel (or skip)
                                                  # torch pulls in matching triton etc.

# 3. resolve / install everything else, incl. flash-attn
uv sync                                           # uses pyproject, honours no-build-isolation
