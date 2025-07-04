#!/bin/bash
#SBATCH -J aEIUU_1iter
#SBATCH -o /home/asj53/aEIUU_%j.out
#SBATCH -e /home/asj53/aEIUU_%j.err
#SBATCH --partition=frazier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH -t 23:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/asj53/BOScheduling/optimization

set -x
source ~/.bashrc
conda activate research_env

# Pass in the prefs file and override n_iterations
python -u run_EIUU.py \
    --prefs /home/asj53/BOScheduling/results/fa24/pending_prefs.json \
    --n_iterations 1
