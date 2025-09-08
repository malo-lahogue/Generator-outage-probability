import pandas as pd
import numpy as np

# Mutual information
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from npeet import entropy_estimators as ee

from sklearn.preprocessing import StandardScaler

import inferenceModelsV2 as im
import multiprocessing as mp

import time, os


weather_data_file = 'DATA/weather_state_day_enhanced.csv'
power_load_file = 'DATA/power_load_input.csv'
weather_data = pd.read_csv(weather_data_file, index_col=[0,1], parse_dates=[0])
power_load_data = pd.read_csv(power_load_file, index_col=[0,1], parse_dates=[0])


# features_names=['PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMIN', 'TMAX', 'ASTP', 'AWND', 
#                 'PAVG', 'PMIN', 'PMAX', 'PDMAX',
#                 'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']
features_names = list(weather_data.columns) + list(power_load_data.columns) + ['Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']

merged_count_df, feature_names, target_columns = im.preprocess_data(failure_path='DATA/filtered_events.csv',
                                                                event_count_path='DATA/event_count.csv',
                                                                weather_path=weather_data_file,
                                                                power_load_path=power_load_file,
                                                                feature_names=features_names,
                                                                target='Unit_Failure',  # 'Frequency' or 'Unit_Failure'
                                                                state_one_hot=False,
                                                                cause_code_n_clusters=1,
                                                                feature_na_drop_threshold=0.2)                                                                

discrete_features = ['C_0', 'Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']+[f for f in feature_names if f.startswith('State')]
merged_count_df[discrete_features] = merged_count_df[discrete_features].astype('int')

stand_cols = [f for f in feature_names if not f.startswith('State_') and not f in ['Holiday', 'Weekend']]





def _as_cont2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a

def _H_from_counts(counts: np.ndarray, base=np.e) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(base))

def _H_disc(series: pd.Series) -> float:
    vc = series.value_counts(dropna=False)
    return _H_from_counts(vc.to_numpy())

def _disc_keyframe(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Pack multiple discrete cols into one label series for counting (no hash collisions)."""
    if not cols:
        # single dummy key: all rows in one group
        return pd.Series(["__ALL__"] * len(df), index=df.index)
    return df[cols].astype("string").agg("|".join, axis=1)

def _build_cont_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols: 
        return np.empty((len(df), 0))
    mats = [df[c].to_numpy(dtype=float, copy=False).reshape(-1,1) for c in cols]
    return np.hstack(mats)

def _mi_left_vs_right(
    df: pd.DataFrame,
    left_cont_cols: list[str],    # continuous block on the left
    left_disc_cols: list[str],    # discrete block on the left
    right_col: str,               # Y (one feature)
    right_is_discrete: bool,
    k: int = 5
) -> float:
    """
    Hybrid MI: MI([left_cont,left_disc]; Y)
      If Y is discrete:
         MI = E_S[ MI(left_cont ; Y | S) ] + MI(S ; Y)   where S = left_disc
         - E_S term via NPEET micd with adaptive k per stratum
         - MI(S;Y) via discrete counts
      If Y is continuous:
         MI = E_S[ MI(left_cont ; Y | S) ] + MI(S ; Y)
         - E_S term via NPEET mi with adaptive k per stratum
         - MI(S;Y) via NPEET midc (discrete–continuous)
    """
    d = df[left_cont_cols + left_disc_cols + [right_col]].dropna().copy()
    n = len(d)
    if n == 0:
        return 0.0

    # ----- Part 1: E_S[ MI(left_cont ; Y | S) ] -----
    mi_cont = 0.0
    if left_disc_cols:
        groups = d.groupby(left_disc_cols, dropna=False, sort=False)
    else:
        # one "group" containing all rows
        groups = [(None, d)]

    for _, g in groups:
        m = len(g)
        if m <= 1:
            continue
        Xc = _build_cont_matrix(g, left_cont_cols)  # (m, dc)
        if Xc.shape[1] == 0:
            continue  # no continuous contribution in this stratum

        if right_is_discrete:
            y = g[right_col].astype(int).to_numpy()
            # adaptive k_i bounded by smallest class size - 1
            cls_counts = np.bincount(y) if y.ndim == 1 else np.bincount(y.ravel())
            min_cls = cls_counts[cls_counts > 0].min() if cls_counts.sum() > 0 else 0
            k_i = min(k, m - 1, max(min_cls - 1, 0))
            if k_i >= 1:
                mi_cont += (m / n) * float(ee.micd(Xc, y.reshape(-1,1), k=k_i))
        else:
            y = _as_cont2d(g[right_col].to_numpy())
            k_i = min(k, m - 1)
            if k_i >= 1:
                mi_cont += (m / n) * float(ee.mi(Xc, y, k=k_i))

    # ----- Part 2: MI(S ; Y) -----
    if left_disc_cols:
        if right_is_discrete:
            # I(S;Y) = H(S)+H(Y)-H(S,Y)
            Skey = _disc_keyframe(d, left_disc_cols)
            H_S  = _H_disc(Skey)
            H_Y  = _H_disc(d[right_col].astype(int))
            H_SY = _H_disc(Skey + "||" + d[right_col].astype(int).astype(str))
            mi_disc = H_S + H_Y - H_SY
        else:
            # discrete–continuous MI via NPEET midc
            codes_2d = _disc_keyframe(d, left_disc_cols).astype("category").cat.codes.to_numpy().reshape(-1, 1)
            k_dc = min(k, len(d) - 1)
            mi_disc = float(ee.midc(codes_2d, _as_cont2d(d[right_col].to_numpy()), k=k_dc)) if k_dc >= 1 else 0.0
    else:
        mi_disc = 0.0

    return mi_cont + mi_disc

def cmi_hybrid_mixed_XY(
    df: pd.DataFrame,
    X_col: str,            # <-- your binary target
    Y_col: str,            # <-- one candidate feature (discrete or continuous)
    Z_cols: list[str],     # <-- set of already selected features (mixed, can be empty)
    discrete_features: set[str],
    k: int = 5
) -> float:
    """
    Compute I(X;Y|Z) for mixed data using a CMIh-style hybrid:
       I(X;Y|Z) = MI([X,Z]; Y) - MI(Z; Y)
    X is discrete (binary). Y can be discrete or continuous. Z is mixed (possibly empty).
    """
    used = [X_col, Y_col] + list(Z_cols)
    d = df[used].dropna().copy()
    if len(d) == 0:
        return 0.0

    # Partition left side(s) for each MI call into continuous vs discrete
    # Left for MI([X,Z];Y):
    XZ_disc = [c for c in ([X_col] + Z_cols) if c in discrete_features]
    XZ_cont = [c for c in ([X_col] + Z_cols) if c not in discrete_features]
    # Left for MI(Z;Y):
    Z_disc  = [c for c in Z_cols if c in discrete_features]
    Z_cont  = [c for c in Z_cols if c not in discrete_features]

    Y_is_disc = (Y_col in discrete_features)

    i_xz_y = _mi_left_vs_right(d, XZ_cont, XZ_disc, Y_col, right_is_discrete=Y_is_disc, k=k)
    i_z_y  = _mi_left_vs_right(d,  Z_cont,  Z_disc, Y_col, right_is_discrete=Y_is_disc, k=k) if (Z_cont or Z_disc) else 0.0
    return float(i_xz_y - i_z_y)




def get_mutual_information(df: pd.DataFrame,
                          feature_name: str,
                          target_col: str,
                          feature_is_discrete: bool,
                          k: int = 3) -> pd.DataFrame:
    """
    Compute mutual information between the target and the feature.
    """
    if feature_is_discrete:
        mi = ee.midd(
            df[feature_name].astype(int).to_numpy(),
            df[target_col].astype(int).to_numpy()
        )
    else:
        mi = ee.micd(
            _as_cont2d(df[feature_name]),
            _as_cont2d(df[target_col]),
            k=k
        )

    return mi



# Build a quick membership set for discrete features
discrete_set = set(discrete_features)

# Optionally subsample for speed on huge n (keeps the same subset each iteration)
# Comment out if you want to use all rows.
MAX_N =7_000_000
if len(merged_count_df) > MAX_N:
    # stratify on C_0 if it’s highly imbalanced; here we do a simple head sample for clarity
    data_short = merged_count_df.sample(n=MAX_N, random_state=42)
else:
    data_short = merged_count_df

# data_short = merged_count_df

# Z standardization
# scaler = StandardScaler()
max_vals, min_vals = data_short[stand_cols].max(), data_short[stand_cols].min()
data_short.loc[:,stand_cols] = (data_short[stand_cols] - min_vals) / (max_vals - min_vals)


ts = time.time()
##############################################
########## conditional Mutual information ################
##############################################

##############################################
###### Conditional Mutual information ########
##############################################
    
kept_features = []
features_cmi = {'rank': [], 'feature': [], 'cmi': []}
k_val = 3
# kept_features = ['TMIN']
# features_cmi = {'rank': [1], 'feature': ['TMIN'], 'cmi': [0.4255530836480954]}

folder_name = f"Results/conditional_mutual_information_k={k_val}_standardized_{str(MAX_N//1000)+'K' if MAX_N<1e6 else str(MAX_N//1e6)+'M'}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


def process_one_feature(params):
    data_short, f, kept, discrete_set = params
    if kept:
        val = cmi_hybrid_mixed_XY(
                    df=data_short,
                    X_col="C_0",               # <-- your binary target is X
                    Y_col=f,                   # <-- candidate feature
                    Z_cols=kept,               # <-- already selected (can be empty)
                    discrete_features=discrete_set,
                    k=k_val
                )
        return (f, val)
    else:
        # no features selected yet, so we compute MI instead of CMI
        feature_is_discrete = f in discrete_set
        val = get_mutual_information(
            df=data_short,
            feature_name=f,
            target_col="C_0",
            feature_is_discrete=feature_is_discrete,
            k=k_val
        )
        return (f, val)

for i,r in enumerate(np.arange(len(feature_names))):  #range(4):#
    scores = {}
    candidates = list(set(feature_names) - set(kept_features))

    with mp.Pool(10) as pool:
        params = [(data_short, f, kept_features, discrete_set) for f in candidates]
        results = pool.map(process_one_feature, params)
    for f, val in results:
        scores[f] = val
    
    # pick best, record, and continue
    best_f = max(scores, key=scores.get)
    kept_features.append(best_f)
    features_cmi['rank'].append(i + 1)
    features_cmi['feature'].append(best_f)
    features_cmi['cmi'].append(scores[best_f])
    te = time.time()

    print(f"Rank {i+1}: {best_f} with {'MI' if len(kept_features) == 1 else 'CMI'} {scores[best_f]:.6f}")
    print(f"Time taken for rank {i+1}: {te - ts:.2f} seconds")
    ts = te
    df_cmi = pd.DataFrame(features_cmi)
    df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['cmi'])
    df_scores.index.name = 'feature'
    df_scores = df_scores.sort_values(by='cmi', ascending=False)
    df_scores.to_csv(f"{folder_name}/conditional_mutual_information_scores_rank_{i+1}.csv")
    df_cmi.to_csv(f"{folder_name}/conditional_mutual_information_ranking.csv", index=False)


# te = time.time()
# print(f"Mutual information and conditional mutual information computed in {te - ts:.2f} seconds.")