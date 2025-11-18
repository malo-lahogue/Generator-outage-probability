#!/bin/bash 
#SBATCH -p mit_preemptable
#####SBATCH -p mit_normal
#####SBATCH -p mit_normal_gpu
#####SBATCH -p mit_quicktest
#SBATCH -c 5
#SBATCH --gpus-per-node=1
####SBATCH --exclusive
#SBATCH --mem 200G
#####SBATCH --time=24:00:00 
#SBATCH --time=6:00:00 
#SBATCH --output=./logs/log%A_%a.out
#SBATCH --error=./logs/error%A_%a.err
#####SBATCH -o ./logs/log%A_%a.txt
#####SBATCH -e ./logs/error%A_%a.txt

module load miniforge

source $HOME/venvs/Load_shedding_surrogate_ENV/bin/activate


# python main_mutual_information.py --k_knn 30 --library "npeet" #"npeet" "sklearn"
# python conditional_mutual_information.py
python main_grid_search.py --models mlp --technologies thermal --initial_state A --states California --device cuda
# python main_train_model.py --models xgb --technologies thermal --initial_state A --states California --device cuda

# California
# New_York
# Texas
# Connecticut