#!/bin/bash 
#SBATCH -p mit_preemptable
#####SBATCH -p mit_normal
######SBATCH -p mit_normal_gpu
#####SBATCH -p mit_quicktest
#SBATCH -c 5
#SBATCH --gpus-per-node=1
####SBATCH --exclusive
#SBATCH --mem 500G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/log%A_%a.out
#SBATCH --error=./logs/error%A_%a.err
#####SBATCH -o ./logs/log%A_%a.txt
#####SBATCH -e ./logs/error%A_%a.txt

module load miniforge

source $HOME/venvs/Load_shedding_surrogate_ENV/bin/activate



python main_GAM.py


# python main_mutual_information.py --k_knn 30 --library "npeet" #"npeet" "sklearn"
# python conditional_mutual_information.py
# python main_grid_search.py --models mlp --technologies thermal --initial_state A  --final_state all --states New_York --device cuda --num_folds_cv 10
# python main_train_model.py --models mlp --technologies thermal --initial_state A --final_state all --states New_York --device cuda
# 
# California
# New_York
# Texas
# Connecticut


# fl length = 10 epochs
# preemptable :  A all (7048777)
# fl length = 5 epochs
# preemptable :  A all (7049205)
# fl length = 20 epochs
# preemptable :  A all con-lin (7049124), A all exp-cos (7049007)
# fl no schedule
# normal :  A all ()
# ce
# normal :  A all ()