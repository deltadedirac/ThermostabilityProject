#!/bin/bash
#SBATCH --job-name=ESM2_Meltome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH -p high
#SBATCH --time=2-00:00:00

papermill "inverse_folding.ipynb" "results_PDB_Download_MeltomeSplits.ipynb" --log-output --log-level DEBUG --progress-bar
