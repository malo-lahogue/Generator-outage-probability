#!/bin/bash 
#SBATCH -p mit_preemptable
#####SBATCH -p mit_normal
#####SBATCH -p mit_normal_gpu
#####SBATCH -p mit_quicktest
#SBATCH -c 5
#SBATCH --gpus-per-node=1
####SBATCH --exclusive
#SBATCH --mem 50G
#####SBATCH --time=06:00:00 
#SBATCH -o ./logs/log%A_%a.txt
#SBATCH -e ./logs/error%A_%a.txt

module load miniforge

source $HOME/venvs/Load_shedding_surrogate_ENV/bin/activate


# python main_mutual_information.py --k_knn 30 --library "npeet" #"npeet" "sklearn"
# python conditional_mutual_information.py
python main_grid_search.py --models xgb
# python main_train_model.py --models mlp --device cpu
