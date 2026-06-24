#!/bin/bash

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit

source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat_clean

export NVIDIA_LIB=/usr/lib/x86_64-linux-gnu

export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:$NVIDIA_LIB:$TORCH_LIB

export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/10_nvidia_535_288_01.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

export EGL_PLATFORM=device
export CUDA_VISIBLE_DEVICES=0
export MAGNUM_GPU_DEVICE=0

export HABITAT_ENV_DEBUG=1

unset DISPLAY

python -m ascent.run \
  habitat_baselines.num_environments=1 \
  habitat.simulator.habitat_sim_v0.gpu_device_id=0
