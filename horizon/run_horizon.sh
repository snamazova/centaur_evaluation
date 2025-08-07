#!/bin/bash
#SBATCH --job-name=llama_3b_horizon
#SBATCH --partition=A100-40GB
#SBATCH --time=01:00:00
#SBATCH --mem=300gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gpu-bind=single:1
#SBATCH --mail-user=sana04@dfki.de
#SBATCH --mail-type=ALL

source $HOME/.bashrc

# Run everything inside the container
srun \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-workdir="$PWD" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"$(dirname "$PWD")":"$(dirname "$PWD")" \
 \
  bash -c "
    echo 'Activating virtual environment' &&
    source ../.env/bin/activate &&
    echo 'Running script...' &&
    python horizon_llama.py \
  "
