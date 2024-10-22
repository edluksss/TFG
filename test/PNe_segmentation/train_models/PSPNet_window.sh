#!/bin/bash
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1

echo "Restoring modules"
module restore
echo "Loading modules"
module load cesga/system pytorch/2.0.0-cuda
echo "Activating conda environment"
export CONDA_ENVS_PATH=$STORE/conda/envs
export CONDA_PKGS_DIRS=$STORE/conda/pkgs
source activate TFG_env
echo "Executing code"

python PSPNet_window.py