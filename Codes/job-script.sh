#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gpus-per-node=2
#SBATCH --mem=8000M               # mémoire par nœud
#SBATCH --time=15:00:00
#SBATCH --mail-user=<mail@gmail.com>
#SBATCH --mail-type=ALL
#SBATCH --account=def-superviseur

cd /home/mzough/projects/def-superviseur/mzough/meningiome
source ../stage_venv/bin/activate

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
python Z_trainvf.py
 
