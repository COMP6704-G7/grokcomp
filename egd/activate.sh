#!/usr/bin/env bash
# Activate the project environment
source /home/wgs/anaconda3/etc/profile.d/conda.sh
conda activate grok
export PYTHONNOUSERSITE=1
echo "Activated grok @ $(which python)"
python -c "import torch, jax; print(f'torch cuda={torch.cuda.is_available()} | jax devices={jax.devices()}')"
