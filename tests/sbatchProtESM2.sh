#!/bin/bash
#SBATCH --job-name=ESM2_Meltome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH -p high
#SBATCH --gres=gpu:1 -w pika
#SBATCH --time=2-00:00:00

papermill "ESM2Facebook_repro_no_fine_tunning.ipynb" "results_ESM2Facebook_embeddings_Meltome.ipynb" --log-output --log-level DEBUG --progress-bar
