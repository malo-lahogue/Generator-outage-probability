#!/bin/bash 
####SBATCH -p mit_preemptable
#SBATCH -p mit_normal
####SBATCH -p mit_normal_gpu
#####BATCH -p mit_quicktest
#SBATCH -c 10
#####SBATCH --gpus-per-node=1
#### SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --time=06:00:00 
#SBATCH -o ./logs/log%A_%a.txt
#SBATCH -e ./logs/error%A_%a.txt

module load miniforge

source $HOME/venvs/Load_shedding_surrogate_ENV/bin/activate


python get_mutual_information.py
