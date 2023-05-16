#!/bin/bash
#SBATCH -A def-kgroling
#SBATCH --time 0-02:00:00
#SBARCH --n-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=v100:1


# load modules and activate env
module load python
module load scipy-stack
source $HOME/mdi-optuna/bin/activate


python train_kaggle.py
