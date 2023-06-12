#!/bin/bash
#SBATCH --job-name=InverseFolding_ESM2_Meltome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH -p urgent
#SBATCH --gres=shard:20 -w pika
#SBATCH --time=2-00:00:00

papermill "ESM2_and_ESM-IF1.ipynb" "results/results_ESM2_and_ESM-IF1_Meltome.ipynb" --log-output --log-level DEBUG --progress-bar
#python ESM2_and_ESM_IF1.py