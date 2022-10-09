#!/bin/bash
#
#SBATCH --job-name=usrnet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=70GB
#SBATCH --gres=gpu:rtx8000:1
#
#SBATCH --mail-type=END
#SBATCH --mail-user=cc6858@nyu.edu

cd /scratch/$USER/ece57000/USRNet_pytorch
python usrnet_train.py