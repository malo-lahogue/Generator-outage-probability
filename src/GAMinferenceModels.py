# Import libraries

# Data processing and manipulation
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch


# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from pygam import LogisticGAM, s

import preprocess_data as ppd


# Typing
from typing import Iterable, Any, Tuple, Dict



def regional_classifier(train_data:pd.DataFrame, region:str, classifier_feats:list):
    regional_classifier_train_data = train_data.copy()
    regional_classifier_train_data['is_region'] = (regional_classifier_train_data['State'] == region).astype(int)

    X_train = regional_classifier_train_data[classifier_feats]
    y_train = regional_classifier_train_data['is_region']

    classifier = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",  # IMPORTANT if target region is rare
            )

    classifier.fit(X_train, y_train)

    return classifier

def regional_weights_function(train_data:pd.DataFrame, region:str, classifier_features:list):
    """
    Compute regional weights for the given region and classifier features.
    """
    train_data = train_data.copy()
    if 'is_region' not in train_data.columns:
        train_data['is_region'] = (train_data['State'] == region).astype(int)
    cls = regional_classifier(train_data, region, classifier_features)

    n_total = len(train_data)
    n_target = np.sum(train_data['is_region'])

    def weight_function(X:pd.DataFrame) -> np.array:
        weights = cls.predict_proba(X[classifier_features])[:, 1]*n_total/n_target
        return weights

    return weight_function

def get_data_splits(all_data_df, 
                    state, 
                    test_frac=0.2, 
                    seed=42):
    
    np.random.seed(seed)

    ######################################################
    ########### Data Split for Regional Model ############
    ######################################################
    # ROI = Region of Interest, ROW = Rest of World
    # Test = test_frac of data from ROI (ensuring test_frac of each (start,end) state)
    # Train = all data from ROW + (1 - test_frac) of data from ROI

    # state_mask = all_data_df[f"State_{state}"] == 1
    state_mask = all_data_df["State"] == state
    train_df = all_data_df[~state_mask].copy().reset_index(drop=True)
    test_df = all_data_df[state_mask].copy().reset_index(drop=True)
    
    N = len(test_df)
    n_test = int(N*test_frac)
    # test_idx = np.random.choice(test_df.index, size=n_test, replace=False)
    test_idx = []
    for (start_state, end_state), group in test_df.groupby(['Initial_gen_state', 'Final_gen_state']):
        N_group = len(group)
        if N_group < 20:
            continue  # skip small groups to avoid issues
        n_test_group = int(N_group * test_frac)
        n_test_group = max(n_test_group, 20)  # ensure at least 20 samples if group is non-empty
        group_test_idx = np.random.choice(group.index, size=n_test_group, replace=False)
        test_idx.append(group_test_idx)
    test_idx = np.concatenate(test_idx)
    train_idx = test_df.index.difference(test_idx)
    if set(train_idx).intersection(set(test_idx)):
        raise ValueError("Train and test indices overlap!")
    train_df = pd.concat([train_df, test_df.iloc[train_idx]]).copy().reset_index(drop=True)
    test_df = test_df.iloc[test_idx].copy().reset_index(drop=True)

    ######################################################
    ########### Feature Engineering ######################
    ######################################################

    # compute load cdf
    train_df['Load_CDF'] = ppd.load_cdf(train_df, train_df)
    test_df['Load_CDF'] = ppd.load_cdf(train_df, test_df)

    # stress modes
    T_nom = 25  # Nominal temperature in Celsius
    L_rated = 1.0  # Rated load in per unit (example value)

    #    train data
    temp_tr = train_df['Temperature'].values
    humid_tr = train_df['Relative_humidity'].values
    load_tr = train_df['Load_CDF'].values


    psi1_tr = therm_load_stress(temp_tr, load_tr, T_nom=T_nom, L_rated=L_rated)
    psi2_tr = cooling_stress(temp_tr, humid_tr)
    psi3_tr = train_df['Temperature_3Dsum_hot'].values
    psi4_tr = train_df['Temperature_3Dsum_cold'].values

    train_df['psi1'] = psi1_tr
    train_df['psi2'] = psi2_tr
    train_df['psi3'] = psi3_tr
    train_df['psi4'] = psi4_tr
    train_df['Stress'] = composit_stress([psi1_tr, psi2_tr, psi3_tr, psi4_tr], weights=np.ones(4)/4)

    if train_df['Stress'].isna().any():
        raise ValueError(f"NaN values found in 'Stress' for training for state {state}!")

    #    test data
    temp_te = test_df['Temperature'].values
    humid_te = test_df['Relative_humidity'].values
    load_te = test_df['Load_CDF'].values

    psi1_te = therm_load_stress(temp_te, load_te, T_nom=T_nom, L_rated=L_rated)
    psi2_te = cooling_stress(temp_te, humid_te)
    psi3_te = test_df['Temperature_3Dsum_hot'].values
    psi4_te = test_df['Temperature_3Dsum_cold'].values

    test_df['psi1'] = psi1_te
    test_df['psi2'] = psi2_te
    test_df['psi3'] = psi3_te
    test_df['psi4'] = psi4_te
    test_df['Stress'] = composit_stress([psi1_te, psi2_te, psi3_te, psi4_te], weights=np.ones(4)/4)

    if test_df['Stress'].isna().any():
        raise ValueError(f"NaN values found in 'Stress' for testing for state {state}!")

    return train_df, test_df

def train_region_models(
    train_data: pd.DataFrame,
    regions: Iterable[str],
    classifier_features: Iterable[str],
    clipping_quantile: float = 0.95,
    gamma: float = 1.0,
    w_region: bool = True,
    w_stress: bool = True,
    test_frac: float = 0.2,
    verbose: bool = False,):


    transition_models = {}
    test_datasets = {}
    ess_res = {}

    for state in tqdm(regions, desc="Training regional models"):
        if verbose:
            print(f"Training transition model for region: {state}")
        reg_train_data, reg_test_data = get_data_splits(train_data, state, test_frac=test_frac, seed=42)

        w_s = reg_test_data['Stress'].values
        w_max = np.quantile(w_s, 0.95)
        w_clipped = np.clip(w_s, None, w_max)
        w = w_clipped/w_clipped.mean()
        reg_test_data['w_stress'] = w

        test_datasets[state] = reg_test_data
        
        regional_train_data = reg_train_data.copy()
        if w_region:
            regional_train_data['is_region'] = (regional_train_data['State'] == state).astype(int)
            r_w_func = regional_weights_function(regional_train_data, state,
                                                classifier_features=classifier_features)

            regional_train_data["w_region"] = r_w_func(regional_train_data)

            w = regional_train_data['w_region'].values
            w_max = np.quantile(w, 0.95)
            w_clipped = np.clip(w, None, w_max)
            w = w_clipped/w_clipped.mean()
            regional_train_data['w_region_final'] = w
        else:
            regional_train_data['w_region'] = 1.0
            regional_train_data['w_region_final'] = 1.0
            
        model, ess = train_transition_model(regional_train_data, state,
                                                    gamma=gamma,
                                                    clipping_quantile=clipping_quantile,
                                                    w_region=w_region,
                                                    w_stress=w_stress,
                                                    features_stage1=['Temperature', 'Load_CDF', 'psi1', 'psi2', 'psi3', 'psi4'],
                                                    features_stage2=['psi1', 'psi2'],
                                                    target_col='Final_gen_state',
                                                    compressed_data_weight='Data_weight',
                                                    verbose=verbose)

        transition_models[state] = model
        ess_res[state] = ess

    return transition_models, test_datasets, ess_res

def train_transition_model(train_data, region, 
                           gamma = 1.0,
                           clipping_quantile = 0.95,
                           w_region = True,
                           w_stress = True,
                           features_stage1: list = None,
                           features_stage2: list = None,
                           target_col='Final_gen_state',
                           compressed_data_weight='Data_weight',
                           verbose=True,):
    """
    
    """
    train_data = train_data.copy()
    w_raw = np.ones(len(train_data))
    if w_stress:
        # Compute stress metric
        ws = compute_stress_weight(train_data['Stress'].values, gamma=gamma)
        w_raw = w_raw * ws
    if w_region:
        w_raw = w_raw * train_data['w_region'].values
    
    w_max = np.quantile(w_raw, clipping_quantile)
    w_clipped = np.clip(w_raw, None, w_max)
    w = w_clipped/w_clipped.mean()
    train_data['w'] = w
    if verbose:
        print(f"Weight stats: min={w.min()}, max={w.max()}")

    ess = (np.sum(w))**2 / np.sum(w**2)
    if verbose:
        print(f"ESS = {100 * ess / len(train_data)} %")

    stage1_models = {}
    stage2_models = {}
    for initial_state in [0,1,2]:
        if verbose:
            print(f"\nProcessing initial MC state: {initial_state}")
        # ---- Stage 1: fit initial -> failed ----
        fit1 = train_stage1_pygam_logistic(
            train_data=train_data,
            current_state=initial_state,
            features=features_stage1,
            target_col=target_col,
            importance_weight='w',
            compressed_data_weight=compressed_data_weight,
            transform_quantile_space=True,
            n_splines=10,
            spline_order=3,
            lam_grid=np.logspace(-2, 0, 6),
            seed=42,
            objective="auto",
            verbose=verbose,
        )

        stage1_models[initial_state] = fit1

        # ---- Stage 2: fit failed -> specific next state ----
        fit2 = train_stage2_pygam_logistic(
            train_data=train_data,
            current_state=initial_state,
            features=features_stage2,
            target_col=target_col,
            importance_weight= 'w',#'w_region_final' if w_region else None,
            compressed_data_weight=compressed_data_weight,
            transform_quantile_space=True,
            n_splines=10,
            spline_order=3,
            lam_grid=np.logspace(-1, 2, 20),
            seed=42,
            objective="auto",
            verbose=verbose,
        )

        stage2_models[initial_state] = fit2

    prob_models = {'A': probabilistic_two_stage_model(0, stage1_models[0], stage2_models[0]),
                   'D': probabilistic_two_stage_model(1, stage1_models[1], stage2_models[1]),
                   'O': probabilistic_two_stage_model(2, stage1_models[2], stage2_models[2])}

    return prob_models, ess

def train_stage1_pygam_logistic(
    train_data: pd.DataFrame,
    current_state:int,
    features=('Temperature', 'Load_CDF', 'psi1', 'psi2', 'psi3', 'psi4'),
    target_col='Final_gen_state',
    importance_weight='w',
    compressed_data_weight='Data_weight',
    transform_quantile_space: bool = True,
    n_splines=25,
    spline_order=3,
    lam_grid=None,                 # 1D grid => diagonal search; list-of-arrays => full cartesian
    seed=42,
    objective="auto",              # 'auto' (=> UBRE for LogisticGAM), or 'GCV', 'UBRE', 'AIC', 'AICc'
    verbose=True,
):
    df = train_data.loc[train_data['Initial_gen_state'] == current_state].copy()

    # ---- y in {0,1} ----
    y = (df[target_col].to_numpy() != current_state).astype(int)

    # ---- weights ----
    if importance_weight in df.columns:
        w_imp = df[importance_weight].to_numpy(dtype=np.float64)
    else:
        w_imp = np.ones(len(df))
        print("No importance weight found, using uniform weights.")
    if compressed_data_weight in df.columns:
        w_cmp = df[compressed_data_weight].to_numpy(dtype=np.float64)
    else:
        w_cmp = np.ones(len(df))
        print("No compressed data weight found, using uniform weights.")
    w = w_imp * w_cmp
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    # ---- quantile transform (fit on all train_data passed in here) ----
    # If you still want a train/val split to fit transforms only on a subset,
    # keep your split logic; otherwise this is simplest.
    if transform_quantile_space:
        transforms = {}
        rng = np.random.default_rng(seed)
        f = ['Datetime_UTC', 'State'] + list(features)
        for col in f:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame.")
        feature_data = df[f].groupby(['Datetime_UTC', 'State']).first().reset_index()[features].copy()

        for f in features:
            x = feature_data[f].to_numpy()
            order = np.argsort(x)
            xs = x[order]
            u = (np.arange(len(xs)) + 0.5) / len(xs)

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            iso.fit(xs, u)
            transforms[f] = iso

        for f in features:
            df[f] = transforms[f].predict(df[f].to_numpy())
    else:
        transforms = None

    X = df[list(features)].to_numpy(dtype=np.float64)

    # ---- model terms ----
    terms = s(0, n_splines=n_splines, spline_order=spline_order)
    for j in range(1, X.shape[1]):
        terms = terms + s(j, n_splines=n_splines, spline_order=spline_order)

    gam = LogisticGAM(terms)

    # ---- lam grid ----
    if lam_grid is None:
        lam_grid = np.logspace(-3, 2, 8)

    # IMPORTANT:
    # - If you pass 1D lam, pyGAM searches diagonally (same lam for all terms).
    # - If you want full cartesian search, pass: lams = [lam_grid] * n_terms
    #   (n_terms here is number of spline terms, i.e., len(features))
    # lams = [lam_grid] * len(features)
    lams = lam_grid

    # ---- gridsearch chooses best lam by objective (GCV/UBRE/etc.) ----
    gam.gridsearch(X, y, weights=w, lam=lams, objective=objective, progress=verbose)
    # gam.gridsearch(X, y, lam=lams, objective=objective, progress=verbose)


    # best model is now stored in `gam`
    if verbose:
        print("best lam:", gam.lam)
        # gam.statistics_ typically includes the selected objective score
        # (keys vary a bit across versions)
        print("available statistics keys:", list(gam.statistics_.keys()))

    return {
        "features": list(features),
        "transform": transforms,
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "lam": gam.lam,          # note: per-term structure (list of lists)
        "objective": objective,
        "gam": gam,
    }

def train_stage2_pygam_logistic(
    train_data: pd.DataFrame,
    current_state:int,
    features=('psi1', 'psi2'),
    target_col='Final_gen_state',
    importance_weight='w_region_final',
    compressed_data_weight='Data_weight',
    transform_quantile_space: bool = True,
    n_splines=25,
    spline_order=3,
    lam_grid=None,                 # 1D grid => diagonal search; list-of-arrays => full cartesian
    seed=42,
    objective="auto",              # 'auto' (=> UBRE for LogisticGAM), or 'GCV', 'UBRE', 'AIC', 'AICc'
    verbose=True,
):
    df = train_data.loc[(train_data['Initial_gen_state'] == current_state)&(train_data['Final_gen_state'] != current_state)].copy()

    # ---- y in {0,1} ----
    target_state = 1 if current_state == 2 else 2
    y = (df[target_col].to_numpy() == target_state).astype(int)

    # ---- weights ----
    if (importance_weight is not None) and (importance_weight in df.columns):
        w_imp = df[importance_weight].to_numpy(dtype=np.float64)
    else:
        w_imp = np.ones(len(df))
        # print("No importance weight found, using uniform weights.")
    if (compressed_data_weight is not None) and (compressed_data_weight in df.columns):
        w_cmp = df[compressed_data_weight].to_numpy(dtype=np.float64)
    else:
        w_cmp = np.ones(len(df))
        print("No compressed data weight found, using uniform weights.")
    w = w_imp * w_cmp
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    # ---- quantile transform (fit on all train_data passed in here) ----
    # If you still want a train/val split to fit transforms only on a subset,
    # keep your split logic; otherwise this is simplest.
    if transform_quantile_space:
        transforms = {}
        rng = np.random.default_rng(seed)
        feature_data = df[['Datetime_UTC', 'State'] + list(features)].groupby(['Datetime_UTC', 'State']).first().reset_index()[features].copy()

        for f in features:
            x = feature_data[f].to_numpy()
            order = np.argsort(x)
            xs = x[order]
            u = (np.arange(len(xs)) + 0.5) / len(xs)

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            iso.fit(xs, u)
            transforms[f] = iso

        for f in features:
            df[f] = transforms[f].predict(df[f].to_numpy())
    else:
        transforms = None

    X = df[list(features)].to_numpy(dtype=np.float64)

    # ---- model terms ----
    terms = s(0, n_splines=n_splines, spline_order=spline_order)
    for j in range(1, X.shape[1]):
        terms = terms + s(j, n_splines=n_splines, spline_order=spline_order)

    gam = LogisticGAM(terms)

    # ---- lam grid ----
    if lam_grid is None:
        lam_grid = np.logspace(-3, 2, 8)

    # IMPORTANT:
    # - If you pass 1D lam, pyGAM searches diagonally (same lam for all terms).
    # - If you want full cartesian search, pass: lams = [lam_grid] * n_terms
    #   (n_terms here is number of spline terms, i.e., len(features))
    # lams = [lam_grid] * len(features)
    lams = lam_grid

    # ---- gridsearch chooses best lam by objective (GCV/UBRE/etc.) ----
    gam.gridsearch(X, y, weights=w, lam=lams, objective=objective, progress=verbose)
    # gam.gridsearch(X, y, lam=lams, objective=objective, progress=verbose)


    # best model is now stored in `gam`
    if verbose:
        print("best lam:", gam.lam)
        # gam.statistics_ typically includes the selected objective score
        # (keys vary a bit across versions)
        print("available statistics keys:", list(gam.statistics_.keys()))

    return {
        "features": list(features),
        "transform": transforms,
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "lam": gam.lam,          # note: per-term structure (list of lists)
        "objective": objective,
        "gam": gam,
    }

def predict_proba_pygam(fit_dict, df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    transforms = fit_dict.get("transform", None)
    if transforms is not None:
        for f in fit_dict["features"]:
            x = df[f].to_numpy()
            df[f] = transforms[f].predict(x)

    X = df[fit_dict["features"]].to_numpy(dtype=np.float64)
    return np.asarray(fit_dict["gam"].predict_mu(X), dtype=np.float64)

class probabilistic_two_stage_model:
    def __init__(self, initial_state: int, stage1_model: dict, stage2_model: dict):
        self.initial_state = initial_state
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        N = len(df)
        p_final = np.zeros((N, 3), dtype=np.float64)


        # Stage 1: initial -> failed
        p_fail = predict_proba_pygam(self.stage1_model, df)
        # print(f"Stage 1 probs: min={p_fail.min()}, max={p_fail.max()}, mean={p_fail.mean()}")

        # Stage 2: failed -> specific next state
        p_next = predict_proba_pygam(self.stage2_model, df)
        # print(f"Stage 2 probs: min={p_next.min()}, max={p_next.max()}, mean={p_next.mean()}")

        p_final[:, self.initial_state] = 1.0 - p_fail
        target_state = 1 if self.initial_state == 2 else 2
        p_final[:, target_state] = p_fail * p_next
        other_state = 3 - self.initial_state - target_state
        p_final[:, other_state] = p_fail * (1.0 - p_next)

        return p_final

def compute_stress_weight(stress, gamma=0.5):
    """Compute stress weight based on stress metric and gamma parameter."""
    s = np.sort(stress)
    N = len(s)
    cdf = np.arange(1, N + 1) / N
    F = np.interp(stress, s, cdf)
    F=F.clip(0, 1-1e-12)
    weight = (1.01 - F) ** (-gamma)
    return weight

def therm_load_stress(temp: float, load: float, T_nom = 25, L_rated=0.5) -> float:
    """Compute a stress metric based on temperature and power load."""
    # Simple example: stress increases with temperature and load
    stress = (temp - T_nom) * (load / L_rated)  # Normalize temp and load
    return stress

def cooling_stress(temp, humidity, B=0.4) -> float:
    """Compute a stress metric based on temperature and humidity."""
    # Example: stress increases with temperature and humidity
    stress = temp + B * humidity
    return stress

def rolling_hot_temp(temp: np.array,
                    T_nom: float,
                    max_lag_hours: int = 72) -> np.array:
    """
    Add ψ_hot(x_t) = sum_{τ=0}^{max_lag} exp(-τ/τ_half) * max(0, temp_{t-τ} - T_nom)
    as a new column 'psi_hot'.

    df must have columns 'Datetime_UTC' and 'Temperature' on an hourly grid.
    """
    tau_half = max_lag_hours / 2.0

    # excess temperature above threshold
    excess = np.maximum(0.0, temp - T_nom)   # shape (N,)

    # exponential weights for τ = 0..max_lag
    tau = np.arange(max_lag_hours + 1, dtype=float)
    weights = np.exp(-tau / tau_half)

    # causal convolution: each point t sees its past up to max_lag_hours
    psi = np.convolve(excess, weights, mode="full")[: len(excess)]

    return psi

def rolling_cold_temp(temp: np.array,
                    T_nom: float,
                    max_lag_hours: int = 72) -> np.array:
    """
    Add ψ_cold(x_t) = sum_{τ=0}^{max_lag} exp(-τ/τ_half) * max(0, T_nom - temp_{t-τ})
    as a new column 'psi_cold'.

    df must have columns 'Datetime_UTC' and 'Temperature' on an hourly grid.
    """
    tau_half = max_lag_hours / 2.0

    # excess temperature above threshold
    excess = np.maximum(0.0, T_nom - temp)   # shape (N,)

    # exponential weights for τ = 0..max_lag
    tau = np.arange(max_lag_hours + 1, dtype=float)
    weights = np.exp(-tau / tau_half)  

    # causal convolution: each point t sees its past up to max_lag_hours
    psi = np.convolve(excess, weights, mode="full")[: len(excess)]

    return psi

###### Analysis functions ######

def composit_stress(psi_list: Iterable[np.array], weights: Iterable[float] = None) -> np.array:
    """Combine multiple stress metrics into a single composite stress metric."""
    if weights is not None:
        if len(psi_list) != len(weights):
            raise ValueError("Length of psi_list must match length of weights.")
    
    mu_list = [np.mean(psi) for psi in psi_list]
    sigma_list = [np.std(psi) for psi in psi_list]
    # Standardize each psi to have mean 0 and std 1
    psi_list = [(psi - mu) / sigma for psi, mu, sigma in zip(psi_list, mu_list, sigma_list)]
    comp = np.array([w * psi ** 2 for w, psi in zip(weights, psi_list)]).sum(axis=0) if weights is not None else np.sum(psi_list, axis=0)
    comp = np.sqrt(comp)
    return comp


def CE_min(df, features):
    """Returns the probabilities that minimize the cross entropy on the test set (for comparison with model predictions)"""
    df_counts = df[features+['Data_weight']].copy()
    df_counts.rename(columns={'Data_weight': 'count'}, inplace=True)
    counts =  df_counts.groupby(features).sum('count')
    res = df.join(counts, on=features,  how='left')
    res["probability_CE_min"] = res['Data_weight'] / res['count']

    p0 = res.loc[res['Final_gen_state']==0]
    p1 = res.loc[res['Final_gen_state']==1]
    p2 = res.loc[res['Final_gen_state']==2]

    p0 = p0.groupby(['Datetime_UTC']+features).sum()
    p1 = p1.groupby(['Datetime_UTC']+features).sum()
    p2 = p2.groupby(['Datetime_UTC']+features).sum()

    p0 = p0['probability_CE_min']
    p1 = p1['probability_CE_min']
    p2 = p2['probability_CE_min']

    res = res.join(p0, on=['Datetime_UTC']+features, how='left', rsuffix='_0')
    res = res.join(p1, on=['Datetime_UTC']+features, how='left', rsuffix='_1')
    res = res.join(p2, on=['Datetime_UTC']+features, how='left', rsuffix='_2')

    res.fillna(0, inplace=True)

    probs = res[['probability_CE_min_0', 'probability_CE_min_1', 'probability_CE_min_2']].to_numpy()

    return probs


def compute_fractions_by_time(data_start):
    # ensure tz consistency
    data_start = data_start.copy()
    data_start["Datetime_UTC"] = pd.to_datetime(data_start["Datetime_UTC"], utc=True)

    # total weight per datetime
    total = data_start.groupby("Datetime_UTC")["Data_weight"].sum().rename("Total_gen")

    # weight per (datetime, final_state)
    w = (
        data_start.groupby(["Datetime_UTC", "Final_gen_state"])["Data_weight"]
        .sum()
        .unstack("Final_gen_state", fill_value=0.0)
    )

    # ensure all 3 columns exist
    for k in [0, 1, 2]:
        if k not in w.columns:
            w[k] = 0.0
    w = w[[0,1,2]]

    out = pd.concat([total, w], axis=1).reset_index()
    out = out.rename(columns={0:"n0", 1:"n1", 2:"n2"})
    out["f0"] = out["n0"] / out["Total_gen"]
    out["f1"] = out["n1"] / out["Total_gen"]
    out["f2"] = out["n2"] / out["Total_gen"]
    return out  # one row per datetime


def compute_pred_by_time(data_start):
    data_start = data_start.copy()
    data_start["Datetime_UTC"] = pd.to_datetime(data_start["Datetime_UTC"], utc=True)

    w = data_start["Data_weight"]

    pred = (
        data_start.assign(
            wp0=data_start["p0"] * w,
            wp1=data_start["p1"] * w,
            wp2=data_start["p2"] * w,
        )
        .groupby("Datetime_UTC", as_index=False)
        .agg(
            wp0=("wp0", "sum"),
            wp1=("wp1", "sum"),
            wp2=("wp2", "sum"),
            Total_gen=("Data_weight", "sum"),
        )
    )

    # normalize to get weighted averages
    pred["p0"] = pred["wp0"] / pred["Total_gen"]
    pred["p1"] = pred["wp1"] / pred["Total_gen"]
    pred["p2"] = pred["wp2"] / pred["Total_gen"]

    return pred[["Datetime_UTC", "p0", "p1", "p2", "Total_gen"]]


def calibration_curve_equal_count(p, y, w=None, n_bins=20):
    """
    p, y: 1D arrays of same length
    w: optional weights
    Returns: bin_p_mean, bin_y_mean
    """
    p = np.asarray(p)
    y = np.asarray(y)
    assert p.shape == y.shape
    if w is None:
        w = np.ones_like(p, dtype=float)
    else:
        w = np.asarray(w)
        assert w.shape == p.shape

    # sort by predicted prob
    idx = np.argsort(p)
    p = p[idx]; y = y[idx]; w = w[idx]

    N = len(p)
    if N == 0:
        return np.array([]), np.array([])

    # split into n_bins chunks (some may be empty if N < n_bins)
    edges = np.linspace(0, N, n_bins+1).astype(int)

    p_means, y_means = [], []
    for i in range(n_bins):
        a, b = edges[i], edges[i+1]
        if b <= a:
            continue
        pb = p[a:b]; yb = y[a:b]; wb = w[a:b]
        p_means.append(np.average(pb, weights=wb))
        y_means.append(np.average(yb, weights=wb))

    return np.array(p_means), np.array(y_means)


def plot_calibration_matrix(
    full_model_results,
    test_data_stress_bins_per_state,
    n_bins=5,
    init_states=(0, 1, 2),
    end_states=(0, 1, 2),
    figsize=(20, 20),
    cmap=plt.cm.YlOrRd,
    eps=0.1,
    font_size_title=20,
    font_size_label=16,
    font_size_legend=20,
    marker_size=60,
):
    """
    Plot a matrix of calibration plots:
      rows   = initial states
      cols   = end states

    Parameters
    ----------
    full_model_results : dict[state -> DataFrame]
    test_data_stress_bins_per_state : dict[state -> dict[stress_bin -> DataFrame]]
    compute_fractions_by_time : callable
    compute_pred_by_time : callable
    calibration_curve_equal_count : callable
    """

    # --- stress bins & colors (consistent across panels) ---
    example_state = next(iter(test_data_stress_bins_per_state))
    stress_bins = list(test_data_stress_bins_per_state[example_state].keys())
    stress_colors = {
        stress: cmap((i + 1) / (len(stress_bins) + 1))
        for i, stress in enumerate(stress_bins)
    }
    idx2state = {0: "A", 1: "D", 2: "O"}
    nrows, ncols = len(init_states), len(end_states)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    # make axs always 2D
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
    if ncols == 1:
        axs = np.expand_dims(axs, axis=1)

    legend_handles = {}
    perfect_handle = None

    for i, init_state in enumerate(init_states):
        for j, end_state in enumerate(end_states):
            ax = axs[i, j]

            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            pooled = defaultdict(lambda: {"p": [], "y": [], "w": []})

            for state, test_data in full_model_results.items():
                data_start = test_data.loc[
                    test_data["Initial_gen_state"] == init_state
                ].copy()
                if data_start.empty:
                    continue

                truth = compute_fractions_by_time(data_start)
                pred = compute_pred_by_time(data_start)
                cal_df = pred.merge(
                    truth[["Datetime_UTC", "f0", "f1", "f2"]],
                    on="Datetime_UTC",
                    how="inner",
                )
                if cal_df.empty:
                    continue

                stress_bins_state = test_data_stress_bins_per_state[state]

                for bin_name, bin_data in stress_bins_state.items():
                    if "Datetime_UTC" in bin_data.columns:
                        dt_bin = pd.to_datetime(bin_data["Datetime_UTC"], utc=True)
                    else:
                        dt_bin = pd.to_datetime(bin_data.index, utc=True)

                    cal_bin = cal_df[cal_df["Datetime_UTC"].isin(dt_bin)]
                    if cal_bin.empty:
                        continue

                    pooled[bin_name]["p"].extend(cal_bin[f"p{end_state}"])
                    pooled[bin_name]["y"].extend(cal_bin[f"f{end_state}"])
                    pooled[bin_name]["w"].extend(cal_bin["Total_gen"])

            min_v, max_v = 1.0, 0.0
            any_points = False

            for bin_name, d in pooled.items():
                p = np.asarray(d["p"])
                y = np.asarray(d["y"])
                w = np.asarray(d["w"])

                if len(p) == 0:
                    continue

                px, py = calibration_curve_equal_count(
                    p, y, w=w, n_bins=n_bins
                )

                h = ax.scatter(
                    px,
                    py,
                    s=marker_size,
                    color=stress_colors[bin_name],
                    alpha=0.9,
                )

                legend_handles.setdefault(bin_name, h)
                any_points = True

                min_v = min(min_v, px.min(), py.min())
                max_v = max(max_v, px.max(), py.max())

            if not any_points:
                ax.set_axis_off()
                continue

            perfect_handle = ax.plot(
                [min_v, max_v],
                [min_v, max_v],
                linestyle="--",
                linewidth=1.4,
                color="gray",
            )[0]

            delta = (max_v - min_v) * eps
            ax.set_xlim(min_v - delta, max_v + delta)
            ax.set_ylim(min_v - delta, max_v + delta)

            ax.set_title(
                # f"Init {init_state} → End {end_state}",
                f"{idx2state[init_state]} → {idx2state[end_state]}",
                fontsize=font_size_title,
            )

            if j == 0:
                ax.set_ylabel(
                    "Observed frequency",
                    fontsize=font_size_label,
                )
            if i == nrows - 1:
                ax.set_xlabel(
                    "Mean predicted probability",
                    fontsize=font_size_label,
                )
            ax.tick_params(axis="both", which="major", labelsize=font_size_label)
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(font_size_label)
            ax.yaxis.get_offset_text().set_fontsize(font_size_label)

    # --- shared legend ---
    handles = [legend_handles[s] for s in stress_bins if s in legend_handles]
    labels = [s for s in stress_bins if s in legend_handles]

    if perfect_handle is not None:
        handles.append(perfect_handle)
        labels.append("Perfect calibration")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=False,
            fontsize=font_size_legend,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def weighted_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    weights: np.ndarray,
    n_pts_per_bin: int = 500,
    eps: float = 1e-12,
):
    """
    Weighted relative calibration error using equal-count bins on p_pred.

    Returns
    -------
    (wrce, total_weight)

    Notes
    -----
    - Bins are defined by quantiles of p_pred (approximately equal number of points per bin).
    - Per-bin calibration error: |p_bar - y_bar| / denom
      where denom is y_bar (relative_to="y") or p_bar (relative_to="p"), stabilized by eps.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # basic checks
    if not (len(y_true) == len(p_pred) == len(weights)):
        raise ValueError("y_true, p_pred, weights must have the same length")
    if len(y_true) == 0:
        return np.nan, 0.0

    n_bins = max(1, len(y_true) // n_pts_per_bin)

    # quantile edges (may have duplicates if p_pred has many ties)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(p_pred, qs, method='inverted_cdf', weights=weights)
    bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)  # include max safely

    CE = []
    P = []
    W = []

    for i in range(n_bins):
        s, e = bin_edges[i], bin_edges[i + 1]
        mask = (p_pred >= s) & (p_pred < e)

        if not np.any(mask):
            continue

        y = y_true[mask]
        p = p_pred[mask]
        w = weights[mask]

        wsum = np.sum(w)
        if wsum <= 0:
            continue

        p_bar = np.average(p, weights=w)
        y_bar = np.average(y, weights=w)


        ce = abs(p_bar - y_bar)
        CE.append(ce)
        P.append(p_bar)
        W.append(wsum)

    if len(W) == 0:
        return np.nan, 0.0

    return CE, W, P


def weighted_log_likelihood(y_true: np.ndarray, p_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted log-likelihood for multi-class classification."""
    eps = 1e-12
    p_pred = np.clip(p_pred, eps, 1 - eps)
    log_likelihoods = np.log(p_pred[np.arange(len(y_true)), y_true])
    weighted_log_likelihood = np.sum(weights * log_likelihoods) / np.sum(weights)
    return weighted_log_likelihood

def weighted_brier_score(
    n: np.ndarray,
    N: np.ndarray,
    p_pred: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-12):

    return np.sum(weights * (n * (1 - p_pred) ** 2)+(N-n)*p_pred**2) / np.sum(weights * N)

def plot_score_by_stress_grouped_bars(
    score_function,
    models_evaluated,
    test_data_stress_bins_per_state,
    stress_bin_order=["Low", "Medium", "High", "Very High"],
    figsize=(11, 5),
    title="Weighted log-likelihood by stress bin",
    ylabel="Weighted log-likelihood",
    fontsize=13,
    ticksize=12,
    legendsize=11,
    rotate_xticks=0,
    value_fmt="{:.2e}",   # you can change formatting here
):
    # best direction
    higher_is_better = (score_function == weighted_log_likelihood)

    model_names = list(models_evaluated.keys())
    n_models = len(model_names)
    n_bins = len(stress_bin_order)

    x = np.arange(n_bins)

    # bar geometry
    total_width = 0.82
    bar_w = total_width / max(1, n_models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_w

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors

    # store all scores to find best bars later
    all_scores = np.full((n_models, n_bins), np.nan, dtype=float)
    bar_containers = []

    for i, model_name in enumerate(model_names):
        model = models_evaluated[model_name]
        score_per_bin = []

        for stress_bin in stress_bin_order:
            score_vals = []
            w_vals = []

            for state, _model_data in model.items():
                bin_data = test_data_stress_bins_per_state[model_name][state][stress_bin]

                if score_function == weighted_log_likelihood:
                    y_true = bin_data["Final_gen_state"].to_numpy()
                    p_pred = bin_data[["p0", "p1", "p2"]].to_numpy()
                    w_cp = bin_data["Data_weight"].to_numpy()
                    w_im = bin_data["w_stress"].to_numpy()
                    w = w_cp * w_im

                    score_vals.append(score_function(y_true, p_pred, w))
                    w_vals.append(np.sum(w))

                

                elif score_function == weighted_calibration_error:
                    for initial_state in [0, 1, 2]:
                        df_data = bin_data.loc[bin_data["Initial_gen_state"] == initial_state]
                        if df_data.empty:
                            continue
                        df_data = df_data.drop_duplicates(subset=["Datetime_UTC"])

                        for final_state in [0, 1, 2]:
                            f = df_data[f"f{final_state}"].to_numpy()
                            p = df_data[[f"p{final_state}"]].to_numpy().flatten()
                            w_cp = df_data["Total_gen"].to_numpy()
                            w_im = df_data["w_stress"].to_numpy()
                            w = w_cp * w_im
                            score, total_w, _ = score_function(f, p, w)
                            # score could be vector; total_w could be vector
                            score_vals.extend(score)
                            w_vals.extend(total_w)
                elif score_function == weighted_brier_score:
                    for initial_state in [0, 1, 2]:
                        df_data = bin_data.loc[bin_data["Initial_gen_state"] == initial_state]
                        if df_data.empty:
                            continue
                        df_data = df_data.drop_duplicates(subset=["Datetime_UTC"])
                        for final_state in [0, 1, 2]:
                            n = df_data['Data_weight'].to_numpy()
                            N = df_data['Total_gen'].to_numpy()
                            w = df_data['w_stress'].to_numpy()
                            p_pred = df_data[f"p{final_state}"].to_numpy()

                            bs = score_function(n, N, p_pred, w)
                            score_vals.append(bs)
                            w_vals.append(np.sum(w))

            score_per_bin.append(np.average(score_vals, weights=w_vals) if len(score_vals) else np.nan)

        all_scores[i, :] = score_per_bin

        bars = ax.bar(
            x + offsets[i],
            score_per_bin,
            width=bar_w,
            label=model_name,
            color=colors[i % len(colors)],
            edgecolor="none",   # important: keep legend patches unframed
            linewidth=0.0,
        )
        bar_containers.append(bars)

    # ---------- highlight best bars (black frame on the plot only) ----------
    for b in range(n_bins):
        col = all_scores[:, b]
        valid = np.isfinite(col)
        if not np.any(valid):
            continue
        best_idx = np.argmax(col[valid]) if higher_is_better else np.argmin(col[valid])
        best_idx = np.where(valid)[0][best_idx]

        best_bar = bar_containers[best_idx][b]
        best_bar.set_edgecolor("black")
        best_bar.set_linewidth(2.5)

    # ---------- annotate values, and compute safe y-lims ----------
    finite_vals = all_scores[np.isfinite(all_scores)]
    if finite_vals.size == 0:
        finite_vals = np.array([0.0])

    # scale for text offset
    y_span = float(np.max(finite_vals) - np.min(finite_vals))
    if y_span == 0:
        y_span = max(1.0, abs(float(np.max(finite_vals))))

    # where labels will end up (for ylim padding)
    label_ys = []

    max_label_height = 0.0
    for bars in bar_containers:
        for bar in bars:
            h = bar.get_height()
            if not np.isfinite(h):
                continue

            x_pos = bar.get_x() + bar.get_width() / 2

            # offset is proportional to plot scale (works for both small/large numbers)
            pad = 0.03 * y_span
            if h >= 0:
                y_text = h + pad
                va = "bottom"
            else:
                y_text = h - pad
                va = "top"
            # if y_text > 0 :
            #     max_label_height = max(max_label_height, y_text)
            # else:
            #     max_label_height = min(max_label_height, y_text)

            ax.text(
                x_pos,
                y_text,
                value_fmt.format(h),
                ha="center",
                va=va,
                fontsize=ticksize,
                rotation=90,
                clip_on=False,   # allow text outside axes, we'll expand ylim to include it
            )
            label_ys.append(y_text)

    # Expand y-limits so labels never overlap axes (and aren’t cut off)
    ymin_data, ymax_data = np.min(finite_vals), np.max(finite_vals)
    ymin_label = np.min(label_ys) if label_ys else ymin_data
    ymax_label = np.max(label_ys) if label_ys else ymax_data

    # extra margin so text doesn't touch borders
    margin = 0.8 * y_span
    ax.set_ylim(min(ymin_data, ymin_label) - margin, max(ymax_data, ymax_label) + margin)

    # ---------- formatting ----------
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel("Stress bin", fontsize=fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(stress_bin_order, fontsize=ticksize, rotation=rotate_xticks)
    ax.tick_params(axis="y", labelsize=ticksize)

    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------- Legend: custom handles so model colors are NOT framed ----------
    model_handles = [
        Patch(facecolor=colors[i % len(colors)], edgecolor="none", label=name)
        for i, name in enumerate(model_names)
    ]
    best_handle = Patch(facecolor="white", edgecolor="black", linewidth=2.0, label="Best model")

    handles = model_handles + [best_handle]

    ax.legend(
        handles=handles,
        title="Model",
        fontsize=legendsize,
        title_fontsize=legendsize,
        frameon=True,
        ncol=min(4, len(handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
    )

    # plt.tight_layout()
    plt.show()



