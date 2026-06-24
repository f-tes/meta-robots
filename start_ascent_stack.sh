#!/bin/bash

set -e

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
ENV_NAME=habitat_clean

start_server () {
  SESSION=$1
  CMD=$2
  PORT=$3

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already exists"
  else
    echo "Starting $SESSION on port $PORT..."
    tmux new-session -d -s "$SESSION" "cd $PROJECT && source $CONDA_SH && conda activate $ENV_NAME && \
      export NVIDIA_LIB=/usr/lib/x86_64-linux-gnu && \
      export TORCH_LIB=\$(python -c \"import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\") && \
      export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6 && \
      export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:\$NVIDIA_LIB:\$TORCH_LIB && \
      export CUDA_VISIBLE_DEVICES=0 && \
      unset DISPLAY && \
      $CMD"
  fi

  echo "Waiting for port $PORT..."
  for i in {1..60}; do
    if curl -s "http://localhost:$PORT" >/dev/null 2>&1; then
      echo "$SESSION is up"
      return 0
    fi
    sleep 2
  done

  echo "WARNING: $SESSION did not respond on port $PORT"
}

start_server blip2 "python -m model_api.blip2itm_out --port 13182" 13182
start_server dfine "python -m model_api.dfine_out --port 13186" 13186

echo "Servers started. Running ASCENT..."

cd $PROJECT
source $CONDA_SH
conda activate $ENV_NAME

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

MODE=${1:-case}

if [ "$MODE" = "full" ]; then
    echo "Running FULL val split..."

    python -m ascent.run \
      habitat_baselines.num_environments=1 \
      habitat.simulator.habitat_sim_v0.gpu_device_id=0

else
    echo "Running SINGLE CASE..."

    python -m ascent.run \
      habitat_baselines.num_environments=1 \
      habitat.simulator.habitat_sim_v0.gpu_device_id=0 \
      habitat.dataset.content_scenes='["TEEsavR23oF"]'
fi
