#!/bin/bash
#SBATCH --job-name=horizon_70b
#SBATCH --partition=A100-80GB
#SBATCH --output=logs/chorizon_test_%j.log
#SBATCH --time=6:00:00
#SBATCH --mem=300gb
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=20
#SBATCH --gpu-bind=none
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
    python rl_llama_one.py \
  "
