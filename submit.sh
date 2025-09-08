#!/bin/bash 
#SBATCH -p mit_preemptable
#####SBATCH -p mit_normal
###SBATCH -p mit_normal_gpu
#####SBATCH -p mit_quicktest
#SBATCH -c 10
#SBATCH --gpus-per-node=1
#### SBATCH --exclusive
#SBATCH --mem=200G
####SBATCH --time=06:00:00 
#SBATCH -o ./logs/log%A_%a.txt
#SBATCH -e ./logs/error%A_%a.txt

module load miniforge

source $HOME/venvs/Load_shedding_surrogate_ENV/bin/activate


# python get_mutual_information.py
# python get_conditional_mutual_information.py
python main_grid_search.py
