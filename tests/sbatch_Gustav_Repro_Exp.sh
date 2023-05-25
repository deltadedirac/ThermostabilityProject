#!/bin/bash
#SBATCH --job-name=ESM2_Gustav_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH -p high
#SBATCH --gres=gpu:1 -w pika
#SBATCH --time=2-00:00:00

papermill "Reproduction_Gustav_Thesis_same_Dataset.ipynb" "results/results_ESM2_Gustav_Same_Data.ipynb" --log-output --log-level DEBUG --progress-bar
#python ESM2_and_ESM_IF1.py