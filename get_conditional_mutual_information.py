import pandas as pd
import numpy as np

# Mutual information
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from npeet import entropy_estimators as ee

import inferenceModelsV2 as im
import multiprocessing as mp

import time





features_names=['PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMIN', 'TMAX', 'ASTP', 'AWND', 
                'PAVG', 'PMIN', 'PMAX', 'PDMAX',
                'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']


merged_count_df, feature_names, target_columns = im.preprocess_data(failure_path='DATA/filtered_events.csv',
                                                                event_count_path='DATA/event_count.csv',
                                                                weather_path='DATA/weather_state_day.csv',
                                                                power_load_path='DATA/power_load_input.csv',
                                                                feature_names=features_names,
                                                                target='Unit_Failure',  # 'Frequency' or 'Unit_Failure'
                                                                state_one_hot=False,
                                                                cause_code_n_clusters=1)

discrete_features = ['C_0', 'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']+[f for f in feature_names if f.startswith('State')]
merged_count_df[discrete_features] = merged_count_df[discrete_features].astype('int')

stand_cols = [f for f in feature_names if not f.startswith('State_') and not f in ['Holiday', 'Weekend']]





# --- helpers for NPEET formatting ---
def _to_list2d(col: pd.Series, dtype=float):
    """(n,) -> [[v1],[v2],...] as python lists, dtype-cast."""
    return col.to_numpy(dtype=dtype, copy=False).reshape(-1, 1).tolist()

def _to_list2d_matrix(df: pd.DataFrame, dtype=float):
    """(n,d) -> [[...d...], ...] as python lists, dtype-cast."""
    return df.to_numpy(dtype=dtype, copy=False).tolist()

def _jitter_if_discrete(col: pd.Series, is_discrete: bool, eps=1e-10, seed=0):
    """
    KSG estimators assume continuous vars (no exact ties).
    For discrete vars, add tiny noise to break ties safely.
    """
    if not is_discrete:
        return _to_list2d(col, dtype=float)
    rng = np.random.default_rng(seed)
    arr = col.to_numpy(dtype=float, copy=False)
    arr = arr + rng.normal(0.0, eps, size=arr.shape)
    return arr.reshape(-1, 1).tolist()

# Build a quick membership set for discrete features
discrete_set = set(discrete_features)

# Optionally subsample for speed on huge n (keeps the same subset each iteration)
# Comment out if you want to use all rows.
# MAX_N = 1000_000
# if len(merged_count_df) > MAX_N:
#     # stratify on C_0 if it’s highly imbalanced; here we do a simple head sample for clarity
#     data_short = merged_count_df.sample(n=MAX_N, random_state=42)
# else:
#     data_short = merged_count_df

data_short = merged_count_df

ts = time.time()
##############################################
########## conditional Mutual information ################
##############################################

##############################################
###### Conditional Mutual information ########
##############################################
    
kept_features = ['TMIN']
features_cmi = {'rank': [1], 'feature': ['TMIN'], 'cmi': [0.4255530836480954]}

def process_one_feature(params):
    data_short, f, discrete_set, Z, X = params
    Y = _jitter_if_discrete(data_short[f], is_discrete=(f in discrete_set))
    if Z is None:
        # First iteration: I(X;Y)
        val = ee.mi(X, Y, k=10)
    else:
        # Subsequent iterations: I(X;Y|Z)
        val = ee.cmidd(X, Y, Z)
    return f, float(val)

for i in range(len(feature_names)):
    scores = {}
    candidates = list(set(feature_names) - set(kept_features))
    # Precompute X once (your "target" side of CMI)
    X = _jitter_if_discrete(data_short['C_0'], is_discrete=True)

    # Precompute Z once per iteration (growing set of already selected features)
    if kept_features:
        Z = _to_list2d_matrix(data_short[kept_features], dtype=float)
    else:
        Z = None  # means we'll compute MI(X;Y) in the first round

    with mp.Pool(10) as pool:
        params = [(data_short, f, discrete_set, Z, X) for f in candidates]
        results = pool.map(process_one_feature, params)
    for f, val in results:
        scores[f] = val

    # for f in candidates:
    #     Y = _jitter_if_discrete(data_short[f], is_discrete=(f in discrete_set))
    #     if Z is None:
    #         # First iteration: I(X;Y)
    #         val = ee.mi(X, Y, k=10)
    #     else:
    #         # Subsequent iterations: I(X;Y|Z)
    #         val = ee.cmi(X, Y, Z, k=10)
    #     scores[f] = float(val)

    # pick best, record, and continue
    best_f = max(scores, key=scores.get)
    kept_features.append(best_f)
    features_cmi['rank'].append(i + 1)
    features_cmi['feature'].append(best_f)
    features_cmi['cmi'].append(scores[best_f])
    te = time.time()

    print(f"Rank {i+1}: {best_f} with {'MI' if Z is None else 'CMI'} {scores[best_f]:.6f}")
    print(f"Time taken for rank {i+1}: {te - ts:.2f} seconds")
    ts = te
    df_cmi = pd.DataFrame(features_cmi)

    df_cmi.to_csv('Results/conditional_mutual_information_ranking.csv', index=False)


# te = time.time()
# print(f"Mutual information and conditional mutual information computed in {te - ts:.2f} seconds.")