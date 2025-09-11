import pandas as pd
import numpy as np

# Mutual information
from sklearn.metrics import mutual_info_score
# from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from npeet import entropy_estimators as ee

import inferenceModelsV2 as im

import time

from typing import Iterable
from pathlib import Path
import sys


k_knn = int(sys.argv[1])



weather_data_file = 'DATA/weather_data_per_state_all.csv'
power_load_file = 'DATA/power_load_input.csv'
weather_data = pd.read_csv(weather_data_file, index_col=[0,1], parse_dates=[0])
power_load_data = pd.read_csv(power_load_file, index_col=[0,1], parse_dates=[0])


# features_names=['PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMIN', 'TMAX', 'ASTP', 'AWND', 
#                 'PAVG', 'PMIN', 'PMAX', 'PDMAX',
#                 'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']
features_names = list(weather_data.columns) + list(power_load_data.columns) + ['Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']
features_names = list(set(features_names)-set(['EventStartDT', 'Date', 'PRCP_30dz']))


merged_count_df, feature_names, target_columns = im.preprocess_data(failure_path='DATA/filtered_events.csv',
                                                                event_count_path='DATA/event_count.csv',
                                                                weather_path=weather_data_file,
                                                                power_load_path=power_load_file,
                                                                feature_names=features_names,
                                                                target='Unit_Failure',  # 'Frequency' or 'Unit_Failure'
                                                                state_one_hot=False,
                                                                cause_code_n_clusters=1,
                                                                feature_na_drop_threshold=0.10)

discrete_features = ['C_0', 'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']+[f for f in feature_names if f.startswith('State')]
merged_count_df[discrete_features] = merged_count_df[discrete_features].astype('int')

stand_cols = [f for f in feature_names if not f.startswith('State_') and not f in ['Holiday', 'Weekend']]
merged_count_df[stand_cols] = (merged_count_df[stand_cols] - merged_count_df[stand_cols].mean()) / merged_count_df[stand_cols].std(ddof=0)#.replace(0, 1.0)


def _is_discrete_series(s: pd.Series, discrete_features: set[str]) -> bool:
    if s.name in discrete_features:
        return True
    return pd.api.types.is_integer_dtype(s) or s.nunique(dropna=False) <= max(20, int(0.01*len(s)))

def _as_cont2d(col: pd.Series) -> np.ndarray:
    return col.to_numpy(dtype=float, copy=False).reshape(-1, 1)

def _as_disc1d(col: pd.Series) -> np.ndarray:
    return col.astype(int).to_numpy(copy=False)

def compute_mutual_information_auto(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    discrete_features: Iterable[str] = (),
    k: int = 3,
    use_rows: pd.Index | None = None,
    out_csv: str = "Results/mutual_information_ranking.csv",
    standardize_continuous: bool = False,
    ) -> pd.DataFrame:
    """
    Auto-choose NPEET estimator:
      - disc X, disc Y        -> ee.midd(x1d, y1d)
      - cont X, disc Y        -> ee.micd(X2d, y2d, k)
      - disc X, cont Y        -> ee.micd(Y2d, x2d, k)   # swap args
      - cont X, cont Y        -> ee.mi(X2d, Y2d, k)
    """
    discrete_set = set(discrete_features)
    data = df.loc[use_rows] if use_rows is not None else df

    # Remove rows with any NaNs across used columns
    used_cols = [target_col] + list(feature_names)
    data = data.dropna(subset=used_cols)

    # Target typing + (optional) standardization plan
    y_is_disc = _is_discrete_series(data[target_col], discrete_set)
    if y_is_disc:
        y_disc_1d = _as_disc1d(data[target_col])          # for midd
        y_disc_2d = y_disc_1d.reshape(-1, 1)              # for micd
        y_cont_2d = None
    else:
        y_cont_2d = _as_cont2d(data[target_col])
        y_disc_1d = y_disc_2d = None

    if standardize_continuous:
        cont_cols = [f for f in feature_names if not _is_discrete_series(data[f], discrete_set)]
        if not y_is_disc and target_col not in cont_cols:
            cont_cols.append(target_col)
        if cont_cols:
            mu = data[cont_cols].mean()
            sd = data[cont_cols].std(ddof=0).replace(0, 1.0)
            data.loc[:, cont_cols] = (data[cont_cols] - mu) / sd

    scores = {}
    for f in feature_names:
        x_is_disc = _is_discrete_series(data[f], discrete_set)

        if x_is_disc and y_is_disc:
            val = ee.midd(_as_disc1d(data[f]), y_disc_1d)

        elif (not x_is_disc) and y_is_disc:
            # X continuous, Y discrete
            val = ee.micd(_as_cont2d(data[f]), y_disc_2d, k=k)

        elif x_is_disc and (not y_is_disc):
            # X discrete, Y continuous  -> swap args for micd
            val = ee.micd(y_cont_2d, _as_disc1d(data[f]).reshape(-1,1), k=k)

        else:
            # both continuous
            val = ee.mi(_as_cont2d(data[f]), y_cont_2d, k=k)

        scores[f] = float(val)

    mi_df = (pd.DataFrame({"feature": list(scores.keys()), "mi": list(scores.values())})
             .sort_values("mi", ascending=False, ignore_index=True))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    mi_df.to_csv(out_csv, index=False)
    print(f"Saved MI ranking to {out_csv}")
    return mi_df


ts = time.time()
##############################################
########## Mutual information ################
##############################################

# Make it comparable to your sklearn run (same rows)
n = len(merged_count_df)
train_end = 10000#int(0.8 * n)
train_idx = merged_count_df.index[:]
# k_knn = 3  # match sklearn's default
out_csv_file = f"Results/mutual_information_ranking_k{k_knn}.csv"

mi_df = compute_mutual_information_auto(
    df=merged_count_df,
    feature_names=feature_names,
    target_col="C_0",                     # or your other target
    discrete_features=discrete_features,  # your explicit list (best source of truth)
    k=k_knn,                                  # match sklearn's n_neighbors for fairer comparison
    use_rows=train_idx,
    out_csv=out_csv_file,
    standardize_continuous=False,         # set True if scales vary a lot
)


##############################################
###### Conditional Mutual information ########
##############################################
    

te = time.time()
print(f"Mutual information and conditional mutual information computed in {te - ts:.2f} seconds.")