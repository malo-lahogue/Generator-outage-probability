import pandas as pd
import numpy as np

# Mutual information
from pathlib import Path
import sys
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((THIS_DIR / "../src").resolve()))
import mutual_information as mi

import inferenceModels as im    # your file

import time
import argparse

from typing import Iterable

# ---------------------------- CLI ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mutual information computation."
    )

    # Data paths
    p.add_argument("--k_knn",  type=int, default=3)
    p.add_argument("--library",    type=str, default="npeet", choices=["npeet", "sklearn"],)

    return p.parse_args()

# ---------------------------- Main ------------------------

def main() -> None:

    args = parse_args()         # Arguments
    weather_data_file = '../DATA/weather_data_per_state_all.csv'
    power_load_file = '../DATA/power_load_input.csv'
    weather_data = pd.read_csv(weather_data_file, index_col=[0,1], parse_dates=[0])
    power_load_data = pd.read_csv(power_load_file, index_col=[0,1], parse_dates=[0])


    # features_names=['PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMIN', 'TMAX', 'ASTP', 'AWND', 
    #                 'PAVG', 'PMIN', 'PMAX', 'PDMAX',
    #                 'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']
    features_names = list(weather_data.columns) + list(power_load_data.columns) + ['Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']
    features_names = list(set(features_names)-set(['EventStartDT', 'Date', 'PRCP_30dz']))


    merged_count_df, feature_names, target_columns = im.preprocess_data(failure_path='../DATA/filtered_events.csv',
                                                                    event_count_path='../DATA/event_count.csv',
                                                                    weather_data_path=weather_data_file,
                                                                    power_data_path=power_load_file,
                                                                    feature_names=features_names,
                                                                    target='Unit_Failure',  # 'Frequency' or 'Unit_Failure'
                                                                    state_one_hot=False,
                                                                    cause_code_n_clusters=1,
                                                                    feature_na_drop_threshold=0.10)

    discrete_features = ['C_0', 'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']+[f for f in feature_names if f.startswith('State')]
    merged_count_df[discrete_features] = merged_count_df[discrete_features].astype('int')

    stand_cols = [f for f in feature_names if not f.startswith('State_') and not f in ['Holiday', 'Weekend']]
    merged_count_df[stand_cols] = (merged_count_df[stand_cols] - merged_count_df[stand_cols].mean()) / merged_count_df[stand_cols].std(ddof=0)#.replace(0, 1.0)


    ts = time.time()
    ##############################################
    ########## Mutual information ################
    ##############################################

    # Make it comparable to your sklearn run (same rows)
    n = len(merged_count_df)
    train_end = n #1000#int(0.8 * n)
    train_idx = merged_count_df.index[:train_end]
    # k_knn = 3  # match sklearn's default
    out_csv_file = f"../Results/mutual_information_ranking_{args.library}_k{args.k_knn}.csv"

    mi_df = mi.compute_mutual_information_auto(
        df=merged_count_df,
        library=args.library,                  # 'npeet' or 'sklearn'
        feature_names=feature_names,
        target_col="C_0",                     # or your other target
        discrete_features=discrete_features,  # your explicit list (best source of truth)
        k=args.k_knn,                                  # match sklearn's n_neighbors for fairer comparison
        use_rows=train_idx,
        out_csv=out_csv_file,
        standardize_continuous=False,         # set True if scales vary a lot
    )


    ##############################################
    ###### Conditional Mutual information ########
    ##############################################
        

    te = time.time()
    print(f"Mutual information and conditional mutual information computed in {te - ts:.2f} seconds.")


if __name__ == "__main__":
    main()