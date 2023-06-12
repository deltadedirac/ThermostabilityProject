#!/bin/bash
#SBATCH --job-name=PiFoldIF_ESM2_Meltome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH -p urgent
#SBATCH --gres=shard:20 -w pika
#SBATCH --time=2-00:00:00

papermill "ESM2_and_PiFold.ipynb" "results/results_ESM2_and_PiFold_Meltome.ipynb" --log-output --log-level DEBUG --progress-bar
#python ESM2_and_ESM_IF1.py