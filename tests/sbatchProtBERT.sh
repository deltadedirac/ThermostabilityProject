#!/bin/bash
#SBATCH --job-name=ProtBERT_Meltome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH -p high
#SBATCH --gres=gpu:1 -w pika
#SBATCH --time=2-00:00:00

papermill "ProtBERT_repro_no_fine_tunning.ipynb" "results_ProtBERT_embeddings_Meltome.ipynb" --log-output --log-level DEBUG --progress-bar
