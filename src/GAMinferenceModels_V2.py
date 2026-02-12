# Import libraries

# Data processing and manipulation
from __future__ import annotations
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

from dataclasses import dataclass
from functools import reduce
import operator
import os

import joblib
import time


# Machine learning models
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


from pygam import LogisticGAM, s

import preprocess_data as ppd


# Typing
from typing import Iterable, Any, Tuple, Dict, Optional, Callable








######## Data ##########

def build_region_dataset(
    all_data_df: pd.DataFrame,
    region: str,
    feature_cols: list[str],
    *,
    test_frac: float = 0.2,
    specific_test_periods: list[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    seed: int = 42,
    w_region_consider: bool = True,
    w_stress_consider: bool = True,
    gamma: float = 1.0,
    clipping_quantile: float | None = 0.95,
    regional_classifier_features=[],  # your ROI-vs-ROW model for this region
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1) split (use your current get_data_splits or a simpler version for now)
    train_df, test_df = get_data_splits(all_data_df, region, test_frac=test_frac, seed=seed, specific_test_periods=specific_test_periods)

    # 2) feature engineering (Stress, Load_CDF, AR features...) should happen HERE
    #    so both train/test have the same columns.
    train_df, test_df = add_engineered_features(train_df, test_df, region)

    # 3) compute weights
    print(region, len(train_df), len(test_df))
    train_df, test_df, ess = add_importance_weights(
        train_df, test_df,
        region,
        w_region_consider=w_region_consider,
        w_stress_consider=w_stress_consider,
        gamma=gamma,
        regional_classifier_features=regional_classifier_features,
        clipping_quantile=clipping_quantile
    )

    train_df['w'] = train_df['Data_weight'] * train_df['w_importance']
    test_df['w'] = test_df['Data_weight'] * test_df['w_importance']



    return train_df, test_df, ess



def get_data_splits(all_data_df, 
                    region, 
                    test_frac=0.2,
                    specific_test_periods: list[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                    seed=42):
    
    rng = np.random.default_rng(seed)

    ######################################################
    ########### Data Split for Regional Model ############
    ######################################################
    # ROI = Region of Interest, ROW = Rest of World
    # Test = test_frac of data from ROI (ensuring test_frac of each (start,end) state)
    # Train = all data from ROW + (1 - test_frac) of data from ROI

    # state_mask = all_data_df[f"State_{state}"] == 1
    # state_mask = all_data_df["State"] == region
    # train_df = all_data_df[~state_mask].copy().reset_index(drop=True)
    # test_df = all_data_df[state_mask].copy().reset_index(drop=True)
    
    # N = len(test_df)
    # test_idx = []
    # for (start_state, end_state), group in test_df.groupby(['Initial_gen_state', 'Final_gen_state']):
    #     N_group = len(group)
    #     if N_group < 20:
    #         continue  # skip small groups to avoid issues
    #     n_test_group = int(N_group * test_frac)
    #     n_test_group = max(n_test_group, 20)  # ensure at least 20 samples if group is non-empty
    #     group_test_idx = rng.choice(group.index, size=n_test_group, replace=False)
    #     test_idx.append(group_test_idx)
    # test_idx = np.concatenate(test_idx)
    # train_idx = test_df.index.difference(test_idx)
    # if set(train_idx).intersection(set(test_idx)):
    #     raise ValueError("Train and test indices overlap!")
    # train_df = pd.concat([train_df, test_df.iloc[train_idx]]).copy().reset_index(drop=True)
    # test_df = test_df.iloc[test_idx].copy().reset_index(drop=True)

    # Alternative normal random split
    if specific_test_periods is None:
        train_idx = rng.choice(all_data_df.index, 
                            size=int(len(all_data_df)*(1 - test_frac)), 
                            replace=False)
        test_idx = all_data_df.index.difference(train_idx)
        train_df = all_data_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = all_data_df.iloc[test_idx].copy().reset_index(drop=True)
    else:
        test_masks = []
        for (start_time, end_time) in specific_test_periods:
            mask = (all_data_df['Datetime_UTC'] >= start_time) & (all_data_df['Datetime_UTC'] <= end_time)
            test_masks.append(mask)
        combined_test_mask = reduce(operator.or_, test_masks)
        test_df = all_data_df[combined_test_mask].copy().reset_index(drop=True)
        train_df = all_data_df[~combined_test_mask].copy().reset_index(drop=True)

    return train_df, test_df


def add_engineered_features(train_df, test_df, region):
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
    train_df['Stress'], mu_train, sigma_train = composit_stress([psi1_tr, psi2_tr, psi3_tr, psi4_tr], weights=np.ones(4)/4)

    if train_df['Stress'].isna().any():
        raise ValueError(f"NaN values found in 'Stress' for training for state {region}!")

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
    test_df['Stress'], _, _ = composit_stress([psi1_te, psi2_te, psi3_te, psi4_te], weights=np.ones(4)/4, mu_list=mu_train, sigma_list=sigma_train)

    if test_df['Stress'].isna().any():
        raise ValueError(f"NaN values found in 'Stress' for testing for state {region}!")

    return train_df, test_df


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


def composit_stress(psi_list: Iterable[np.array], weights: Iterable[float] = None, mu_list: Iterable[float] = None, sigma_list: Iterable[float] = None) -> np.array:
    """Combine multiple stress metrics into a single composite stress metric."""
    if weights is not None:
        if len(psi_list) != len(weights):
            raise ValueError("Length of psi_list must match length of weights.")
    
    if mu_list is None or sigma_list is None:
        mu_list = [np.mean(psi) for psi in psi_list]
        sigma_list = [np.std(psi) for psi in psi_list]
    # Standardize each psi to have mean 0 and std 1
    psi_list = [(psi - mu) / sigma for psi, mu, sigma in zip(psi_list, mu_list, sigma_list)]
    comp = np.array([w * psi ** 2 for w, psi in zip(weights, psi_list)]).sum(axis=0) if weights is not None else np.sum(psi_list, axis=0)
    comp = np.sqrt(comp)
    return comp, mu_list, sigma_list


###### Importance weights ##########

def add_importance_weights(train_df, test_df,
        region,
        w_region_consider,
        w_stress_consider,
        gamma,
        regional_classifier_features,
        clipping_quantile
    ):
    ##### Compute region weights #####
    train_df['is_region'] = (train_df['State'] == region).astype(int)
    train_data_classifier = train_df.drop_duplicates(subset=['Datetime_UTC', 'State']).copy()
    region_classifier = regional_classifier(
        train_data=train_data_classifier,
        region=region,
        classifier_feats=regional_classifier_features,
    )

    p_train_region = region_classifier.predict_proba(train_df[regional_classifier_features])[:, 1]
    p_test_region = region_classifier.predict_proba(test_df[regional_classifier_features])[:, 1]
    eps = 1e-6
    p_train_region = np.clip(p_train_region, eps, 1 - eps)
    p_test_region  = np.clip(p_test_region,  eps, 1 - eps)
    n_total = len(train_df)
    n_target = np.sum(train_df['is_region'])

    train_df['region_weight'] = (p_train_region / (n_target / n_total))
    test_df['region_weight'] = (p_test_region / (n_target / n_total))

    ##### Compute stress weights #####
    # already computed Stress feature in add_engineered_features
    ws_tr = train_df['Stress'].values
    ws_tr_4q = np.sort(train_df.drop_duplicates(subset=['Datetime_UTC', 'State'])['Stress'].values)
    cdf = np.arange(1, len(ws_tr_4q) + 1) / len(ws_tr_4q)
    cdf_tr = np.interp(ws_tr, ws_tr_4q, cdf)
    cdf_tr = cdf_tr.clip(0.0, 1.0 - 1e-6)  # avoid division by zero
    w_s_tr = np.power(1.0 - cdf_tr, -gamma)
    train_df['stress_weight'] = w_s_tr

    ws_te = test_df['Stress'].values
    cdf_te = np.interp(ws_te, ws_tr_4q, cdf)
    cdf_te = cdf_te.clip(0.0, 1.0 - 1e-6)  # avoid division by zero
    w_s_te = np.power(1.0 - cdf_te, -gamma)
    test_df['stress_weight'] = w_s_te


    #### Create importance weights ####
    train_df['w_importance'] = 1.0
    test_df['w_importance'] = 1.0
    if w_region_consider:
        train_df['w_importance'] *= train_df['region_weight']
        test_df['w_importance'] *= test_df['region_weight']
    if w_stress_consider:
        train_df['w_importance'] *= train_df['stress_weight']
        test_df['w_importance'] *= test_df['stress_weight']
    # clipping
    if clipping_quantile is not None:
        clip_tr = train_df['w_importance'].quantile(clipping_quantile)
        train_df['w_importance'] = np.minimum(train_df['w_importance'], clip_tr)
        test_df['w_importance'] = np.minimum(test_df['w_importance'], clip_tr)
    
    # stabilizing with unit mean
    m = train_df['w_importance'].mean()
    train_df['w_importance'] /= m
    test_df['w_importance'] /= m

    ess = (train_df['w_importance'].sum())**2 / (np.sum(train_df['w_importance']**2) + 1e-12)


    return train_df, test_df, ess


######## Classifier model ##########


def regional_classifier(
    train_data: pd.DataFrame,
    region: str,
    classifier_feats: list,
    scale_cols: list = None,
    passthrough_cols: list = None,
    state_col: str = "State",
):
    if scale_cols is None:
        scale_cols = ["Temperature", "Relative_humidity", "psi1", "psi2", "psi3", "psi4", "Stress"]
    if passthrough_cols is None:
        passthrough_cols = ["Load_CDF", "month_sin", "month_cos"]

    df = train_data.copy()

    if state_col not in df.columns:
        raise ValueError(f"{state_col} not found. Available columns include: {list(df.columns)[:30]}")


    # remove cols in scale_cols and passthrough_cols that are not in classifier_feats
    scale_cols = [c for c in scale_cols if c in classifier_feats]
    passthrough_cols = [c for c in passthrough_cols if c in classifier_feats]

    X = df[classifier_feats].copy()
    y = df["is_region"].to_numpy()

    preprocess = ColumnTransformer(
        transformers=[
            ("num_scaled", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), [c for c in scale_cols if c in classifier_feats]),

            ("passthrough", "passthrough", [c for c in passthrough_cols if c in classifier_feats]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            # class_weight="balanced",
            class_weight=None
        )),
    ])

    clf.fit(X, y)
    return clf


######## transition models ##########

IDX_A, IDX_D, IDX_O = 0, 1, 2

def make_stage_A_leave(df: pd.DataFrame):
    sub = df[df["Initial_gen_state"] == IDX_A].copy()
    # y=0: stay in A (A->A), y=1: leave A (A->D or A->O)
    sub["y"] = (sub["Final_gen_state"] != IDX_A).astype(int)
    return sub

def make_stage_A_DO(df: pd.DataFrame):
    sub = df[(df["Initial_gen_state"] == IDX_A) & (df["Final_gen_state"].isin([IDX_D, IDX_O]))].copy()
    # y=0: go to D, y=1: go to O (choose your convention and stick to it)
    sub["y"] = (sub["Final_gen_state"] == IDX_O).astype(int)
    return sub

def make_stage_D_leave(df: pd.DataFrame):
    sub = df[df["Initial_gen_state"] == IDX_D].copy()
    # y=0: stay in D, y=1: leave D (must be D->A)
    sub["y"] = (sub["Final_gen_state"] != IDX_D).astype(int)
    return sub

def make_stage_O_leave(df: pd.DataFrame):
    sub = df[df["Initial_gen_state"] == IDX_O].copy()
    # y=0: stay in O, y=1: leave O (must be O->A)
    sub["y"] = (sub["Final_gen_state"] != IDX_O).astype(int)
    return sub


# def fit_binary_model(model, X, y, sample_weight=None):
#     # X: DataFrame or ndarray
#     # y: 1D array
#     # sample_weight: 1D array or None
#     fit_kwargs = {}
#     if sample_weight is not None:
#         # sklearn uses sample_weight; pygam uses weights
#         if "sample_weight" in model.fit.__code__.co_varnames:
#             fit_kwargs["sample_weight"] = sample_weight
#         elif "weights" in model.fit.__code__.co_varnames:
#             fit_kwargs["weights"] = sample_weight
#         else:
#             raise TypeError("Model.fit does not accept sample_weight/weights.")
#     model.fit(X, y, **fit_kwargs)
#     return model


def train_region_transition_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    zscore_cols: list[str],
    *,
    # base_model_factory: Callable[[], Any],
    type_model: str,
    weight_col: str = "w",
    require_both_classes: bool = True,
    verbose: bool = False
):
    """
    Returns dict with keys: A_leave, A_DO, D_leave, O_leave
    Each value is a fitted binary model.
    """
    models = {}
    scalers = {}

    def _fit(stage_df: pd.DataFrame, name: str, lam_grid=None):
        if stage_df.empty:
            models[name] = None
            return
        y = stage_df["y"].to_numpy()
        if require_both_classes and (y.min() == y.max()):
            # can't fit a classifier on a single class; store None
            models[name] = None
            return

        w = stage_df[weight_col].to_numpy() #if weight_col in stage_df.columns else None
        X = stage_df[feature_cols]

        # pygam: pass numpy array
        if type_model == 'GAM':
            m = make_gam_factory_spline_only(
                                            n_features=len(feature_cols),
                                            n_splines=5,# if name=='A_leave' else 5,
                                        )()
        elif type_model == 'LogisticRegression':
            m = make_sklearn_logistic_factory(
                                            feature_cols= feature_cols,
                                            scale_cols=['Temperature', 'psi1', 'psi2', 'psi3', 'psi4'],
                                            passthrough_cols=['Load_CDF']
                                        )()

        # m = base_model_factory()
        # if isinstance(m, LogisticGAM):
        #     X = X.to_numpy()
        #     # rebuild terms for correct dimension if placeholder:
        #     if m.terms.n_terms != X.shape[1]:
        #         terms = sum([s(i) for i in range(X.shape[1])])
        #         m = LogisticGAM(terms)
        if isinstance(m, LogisticGAM):
            X, scaler = feat_scale_transform(X, zscore_cols)
            scalers[name] = scaler
            models[name] = fit_binary_gam_gcv(m, X, y, sample_weight=w, lam_grid=lam_grid, verbose=verbose)
        else:
            m.fit(X, y, clf__sample_weight=w)
            models[name] = m
        # fit_binary_model(m, X, y, sample_weight=w)

    _fit(make_stage_A_leave(train_df), "A_leave", lam_grid=np.logspace(-1, 1, 6))
    _fit(make_stage_A_DO(train_df), "A_DO", lam_grid=np.logspace(0, 2, 10))
    _fit(make_stage_D_leave(train_df), "D_leave", lam_grid=np.logspace(1, 3, 11))
    _fit(make_stage_O_leave(train_df), "O_leave", lam_grid=np.logspace(1, 3, 11))

    return models, scalers


def predict_transition_probs(models: Dict[str, Any], X, scalers: Dict[str, StandardScaler]) -> pd.DataFrame:
    """
    Given fitted stage models and feature matrix X, produce:
    pAA, pAD, pAO, pDA, pDD, pOA, pOO

    Convention:
    - A_leave predicts P(leave A | x)
    - A_DO predicts P(go to O | left A, x)  (so D prob is 1 - that)
    - D_leave predicts P(leave D | x) => must be D->A when leave
    - O_leave predicts P(leave O | x) => must be O->A when leave
    """
    # X may be DataFrame or ndarray; we pass the same to the model.
    def _p1(name):
        m = models.get(name, None)
        sca = scalers.get(name, None)
        if sca is not None:
            Xs = X.copy()
            Xs[sca.feature_names_in_] = sca.transform(X[sca.feature_names_in_])
            X_use = Xs
        else:
            X_use = X
        if m is None:
            # neutral fallback: 0.0 means "never leave", but safer is NaN to catch issues.
            return np.full(len(X_use), np.nan, dtype=float)
        return predict_proba_binary(m, X_use)

    p_leave_A = _p1("A_leave")
    p_O_given_leaveA = _p1("A_DO")

    # If A_DO is None but A_leave exists, default split 50-50 (or nan)
    if np.any(np.isnan(p_O_given_leaveA)) and not np.all(np.isnan(p_leave_A)):
        p_O_given_leaveA = np.where(np.isnan(p_O_given_leaveA), 0.5, p_O_given_leaveA)

    p_leave_D = _p1("D_leave")
    p_leave_O = _p1("O_leave")

    # A row
    pAA = 1.0 - p_leave_A
    pAO = p_leave_A * p_O_given_leaveA
    pAD = p_leave_A * (1.0 - p_O_given_leaveA)

    # D row (no D->O)
    pDA = p_leave_D
    pDD = 1.0 - p_leave_D

    # O row (no O->D)
    pOA = p_leave_O
    pOO = 1.0 - p_leave_O

    out = pd.DataFrame({
        "pAA": pAA, "pAD": pAD, "pAO": pAO,
        "pDA": pDA, "pDD": pDD,
        "pOA": pOA, "pOO": pOO,
    })
    return out

def predict_proba_binary(model, X) -> np.ndarray:
    """
    Returns P(y=1|x) for a fitted model.
    Supports:
      - sklearn estimators / pipelines (predict_proba -> (n,2))
      - pyGAM LogisticGAM (predict_mu or predict_proba -> (n,))
    """


    # ---- pyGAM first (important!) ----
    if hasattr(model, "predict_mu"):
        Xn = np.asarray(X)  # pygam is happiest with ndarray
        return np.asarray(model.predict_mu(Xn)).ravel()

    # ---- sklearn-like predict_proba ----
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        if p.ndim == 2:
            return p[:, 1]
        # some models return 1D proba directly
        return p.ravel()

    # ---- fallback: decision_function -> sigmoid ----
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(Xn)).ravel()
        return 1.0 / (1.0 + np.exp(-z))

    raise TypeError("Model does not support probability prediction.")

def feat_scale_transform(X: pd.DataFrame, zscore_cols: list[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scales specified columns of X using StandardScaler.
    Returns scaled DataFrame and the scaler used.
    """
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[zscore_cols] = scaler.fit_transform(X[zscore_cols])
    return X_scaled, scaler

# -------------------------------
# 2) Base model factory options
# -------------------------------

def make_sklearn_logistic_factory(
    feature_cols: list[str],
    *,
    scale_cols: Optional[list[str]] = None,
    passthrough_cols: Optional[list[str]] = None,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Callable[[], Any]:
    """
    Returns a factory that builds a fresh sklearn Pipeline:
    (impute + (scale subset) + pass-through subset) -> LogisticRegression
    """
    if scale_cols is None:
        # sensible defaults: scale continuous stress/weather; don't scale cyclic/cdf
        scale_cols = [c for c in feature_cols if c not in ["Load_CDF", "month_sin", "month_cos"]]
    if passthrough_cols is None:
        passthrough_cols = [c for c in ["Load_CDF", "month_sin", "month_cos"] if c in feature_cols]

    scale_cols = [c for c in scale_cols if c in feature_cols]
    passthrough_cols = [c for c in passthrough_cols if c in feature_cols]

    def factory():
        preprocess = ColumnTransformer(
            transformers=[
                ("num_scaled", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), scale_cols),
                ("passthrough", "passthrough", passthrough_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        clf = Pipeline([
            ("preprocess", preprocess),
            ("clf", LogisticRegression(
                penalty="l2",
                C=C,
                solver="lbfgs",
                max_iter=max_iter,
                class_weight=None,   # keep as you prefer
            )),
        ])
        return clf

    return factory


def make_pygam_factory(*, lam=0.6, n_splines=10) -> Callable[[], Any]:
    """
    Simple factory for LogisticGAM with smooth terms for all columns.
    NOTE: pygam expects numpy arrays; we'll pass df[cols].values in training.
    """
    def factory():
        # We'll build terms dynamically at fit-time if needed; here assume X is ndarray
        # and create s(i) for each feature index (handled in fit_region if we pass ndarray).
        # For now: create a placeholder with 1 term; we will overwrite below if desired.
        return LogisticGAM(s(0), lam=lam, n_splines=n_splines)
    return factory


@dataclass
class RegionTrainingResult:
    models: Dict[str, Any]        # stage models
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    ess: float


def train_all_region_transition_models(
    all_data_df: pd.DataFrame,
    regions: list[str],
    *,
    feature_cols: list[str],
    zscore_cols: list[str],
    regional_classifier_features: list[str],
    # base_model_factory: Callable[[], Any],
    type_model:str,
    test_frac: float = 0.2,
    seed: int = 42,
    specific_test_periods: dict[str, list[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    w_region_consider: bool = True,
    w_stress_consider: bool = True,
    gamma: float = 1.0,
    clipping_quantile: Optional[float] = 0.95,
    verbose: bool = False,
) -> Tuple[Dict[str, RegionTrainingResult], Dict[str, pd.DataFrame]]:
    """
    Returns:
      - results_by_region: region -> RegionTrainingResult (stage models + data + ess)
      - preds_by_region:   region -> test_df with transition probability columns appended
    """
    results_by_region: Dict[str, RegionTrainingResult] = {}
    # preds_by_region: Dict[str, pd.DataFrame] = {}
    test_set_by_region = {}
    train_set_by_region = {}
    ess_by_region: Dict[str, float] = {}
    scalers_by_region: Dict[str, Dict[str, StandardScaler]] = {}

    # determinism across regions
    # for k, region in enumerate(tqdm(regions, desc="Training regions")):
    for k, region in enumerate(regions):
        seed_k = seed + k  # avoid identical random split each region if you want; set to seed for fixed
        test_period_state = specific_test_periods.get(region, None) if specific_test_periods is not None else None
        train_df, test_df, ess = build_region_dataset(
            all_data_df,
            region=region,
            feature_cols=feature_cols,
            test_frac=test_frac,
            specific_test_periods=test_period_state,
            seed=seed_k,
            w_region_consider=w_region_consider,
            w_stress_consider=w_stress_consider,
            gamma=gamma,
            clipping_quantile=clipping_quantile,
            regional_classifier_features=regional_classifier_features,
        )
        ess_by_region[region] = ess

        # train stage models on training set
        stage_models, stage_scalers = train_region_transition_model(
            train_df=train_df,
            feature_cols=feature_cols,
            zscore_cols=zscore_cols,
            type_model=type_model,
            # base_model_factory=base_model_factory,
            weight_col="w",
            verbose=verbose,
        )
        scalers_by_region[region] = stage_scalers

        # store
        results_by_region[region] = RegionTrainingResult(
            models=stage_models,
            train_df=train_df,
            test_df=test_df,
            ess=ess,
        )

        # predict on test
        # X_te = test_df[feature_cols]
        # P = predict_transition_probs(stage_models, X_te, scalers=stage_scalers)
        # test_out = test_df.reset_index(drop=True).copy()
        # test_out = pd.concat([test_out, P], axis=1)

        # preds_by_region[region] = test_out
        train_set_by_region[region] = train_df
        test_set_by_region[region] = test_df

        # if verbose:
        #     # quick sanity: probability sums per initial state
        #     for s in [IDX_A, IDX_D, IDX_O]:
        #         sub = test_out[test_out["Initial_gen_state"] == s]
        #         if sub.empty:
        #             continue
        #         if s == IDX_A:
        #             sums = (sub["pAA"] + sub["pAD"] + sub["pAO"]).to_numpy()
        #         elif s == IDX_D:
        #             sums = (sub["pDA"] + sub["pDD"]).to_numpy()
        #         else:
        #             sums = (sub["pOA"] + sub["pOO"]).to_numpy()
        #         print(region, "init", s, "sum range:", np.nanmin(sums), np.nanmax(sums))

    return results_by_region, train_set_by_region, test_set_by_region, ess_by_region, scalers_by_region

def make_gam_factory_spline_only(
    n_features: int,
    *,
    n_splines: int = 10,
    max_iter: int = 300,
    lam_grid=None,
):
    if n_features < 1:
        raise ValueError("n_features must be >= 1")

    terms = reduce(operator.add, (s(i, n_splines=n_splines) for i in range(n_features)))

    def factory():
        gam = LogisticGAM(terms, max_iter=max_iter, verbose=False)
        # store lam grid on the model so fit can reuse it (optional)
        gam._lam_grid = lam_grid
        return gam

    return factory

def fit_binary_gam_gcv(model, X, y, sample_weight=None, lam_grid=None, verbose=False):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    if lam_grid is None:
        # very standard choice in GAM literature
        lam_grid = np.logspace(-3, 3, 11)

    if sample_weight is not None:
        model.gridsearch(X, y, weights=sample_weight, lam=lam_grid, progress=verbose)
    else:
        model.gridsearch(X, y, lam=lam_grid, progress=verbose)

    return model


######### Export #########

idx2state = {0: "A", 1: "D", 2: "O"}
state2idx = {"A": 0, "D": 1, "O": 2}

def export_gam_predictions(
    transition_models: dict,
    test_datasets: dict,
    feature_cols: list[str],
    test: bool,
    out_dir: str,
    scalers_by_region: dict,
    *,
    model_name: str,
    cols_export=['Datetime_UTC', 'Stress', 'Initial_gen_state', 'Final_gen_state', 'Data_weight', 'pAD', 'pAO', 'pDA', 'pOA'],
    region_only = True,
    initial_col: str = "Initial_gen_state",
    final_col: str = "Final_gen_state",
):
    os.makedirs(out_dir, exist_ok=True)

    for region_name, df_test in test_datasets.items():
        if region_name not in transition_models:
            print(f"[WARN] No model for region={region_name}. Skipping.")
            continue

        region_result = transition_models[region_name]   # RegionTrainingResult
        stage_models = region_result.models                   # dict: A_leave, A_DO, D_leave, O_leave

        data = df_test.copy()
        if region_only:
            if region_name not in data['State'].unique():
                print(f"Region {region_name} not in inputed data. /n {data['State'].unique()}")
                continue
            data = data.loc[data['State']==region_name].copy().reset_index()


        # initialize output columns
        for col in ["p0", "p1", "p2", "p"]:
            data[col] = np.nan

        # features matrix (keep DataFrame or ndarray; your helper tolerates both)
        X = data[feature_cols]

        # compute transition probabilities for all rows at once
        # -> columns: pAA pAD pAO pDA pDD pOA pOO
        scalers = scalers_by_region.get(region_name, {})
        P = predict_transition_probs(stage_models, X, scalers=scalers)
        data = pd.concat([data, P], axis=1)

        init = data[initial_col].to_numpy(dtype=int)
        final = data[final_col].to_numpy(dtype=int)

        # Fill p0,p1,p2 depending on initial state
        # Initial = A (0): [A,D,O] = [pAA, pAD, pAO]
        mA = init == 0
        data.loc[mA, "p0"] = P.loc[mA, "pAA"].to_numpy()
        data.loc[mA, "p1"] = P.loc[mA, "pAD"].to_numpy()
        data.loc[mA, "p2"] = P.loc[mA, "pAO"].to_numpy()

        # Initial = D (1): [A,D,O] = [pDA, pDD, 0]
        mD = init == 1
        data.loc[mD, "p0"] = P.loc[mD, "pDA"].to_numpy()
        data.loc[mD, "p1"] = P.loc[mD, "pDD"].to_numpy()
        data.loc[mD, "p2"] = 0.0

        # Initial = O (2): [A,D,O] = [pOA, 0, pOO]
        mO = init == 2
        data.loc[mO, "p0"] = P.loc[mO, "pOA"].to_numpy()
        data.loc[mO, "p1"] = 0.0
        data.loc[mO, "p2"] = P.loc[mO, "pOO"].to_numpy()

        # Probability assigned to the realized Final_gen_state
        # (vectorized gather)
        probs = data[["p0", "p1", "p2"]].to_numpy()
        ok = (final >= 0) & (final <= 2)
        data.loc[ok, "p"] = probs[ok, final[ok]]

        # (Optional) sanity check: rows should sum to ~1 for initial=A, D, O
        # You can uncomment this if useful:
        # s = data[["p0","p1","p2"]].sum(axis=1)
        # print(region_name, "sum min/max:", s.min(), s.max())
        type_out = 'test' if test else 'train'
        if cols_export is not None:
            data = data[cols_export]
        out_path = os.path.join(out_dir, f"GAM_{model_name}_results_{type_out}_{region_name}.csv")
        data.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path} | n={len(data)}")



def export_transition_model_bundle(
    filepath: str,
    *,
    transition_models_by_region: Dict[str, Any],
    scalers_by_region: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[list[str]] = None,
    zscore_cols: Optional[list[str]] = None,
    model_type: str = "GAM",
    extra_metadata: Optional[dict] = None,
) -> None:
    """
    Save a lightweight bundle containing:
      - per-region stage models (RegionTrainingResult.models)
      - per-region scaler (if any)
      - metadata

    Parameters
    ----------
    transition_models_by_region:
        typically the object returned by train_all_region_transition_models
        (dict region -> RegionTrainingResult)
    scalers_by_region:
        dict region -> fitted scaler (StandardScaler) or None
    feature_cols, zscore_cols:
        stored for later checks / convenience
    model_type:
        "GAM" or "LR" (or whatever string you want)
    """
    # Extract only the trained estimators (avoid saving train/test dfs)
    models_only: Dict[str, dict] = {}
    for region, region_result in transition_models_by_region.items():
        # region_result is expected to be RegionTrainingResult
        if not hasattr(region_result, "models"):
            raise TypeError(
                f"Expected each region entry to have a `.models` attribute. "
                f"Got type={type(region_result)} for region={region}."
            )
        models_only[region] = region_result.models

    bundle = {
        "bundle_version": 1,
        "created_unix": time.time(),
        "model_type": model_type,
        "feature_cols": feature_cols,
        "zscore_cols": zscore_cols,
        "models_by_region": models_only,
        "scalers_by_region": scalers_by_region,  # may be None
        "extra_metadata": extra_metadata or {},
    }

    joblib.dump(bundle, filepath, compress=3)


def load_transition_model_bundle(
    filepath: str,
) -> Tuple[Dict[str, dict], Optional[Dict[str, Any]], dict]:
    """
    Load a bundle saved by export_transition_model_bundle.

    Returns
    -------
    models_by_region : dict[region -> dict(stage_name -> model)]
    scalers_by_region: dict[region -> scaler] or None
    metadata         : dict
    """
    bundle = joblib.load(filepath)

    if not isinstance(bundle, dict) or "models_by_region" not in bundle:
        raise ValueError("Not a valid model bundle file (missing 'models_by_region').")

    models_by_region = bundle["models_by_region"]
    scalers_by_region = bundle.get("scalers_by_region", None)
    metadata = {
        "bundle_version": bundle.get("bundle_version"),
        "created_unix": bundle.get("created_unix"),
        "model_type": bundle.get("model_type"),
        "feature_cols": bundle.get("feature_cols"),
        "zscore_cols": bundle.get("zscore_cols"),
        "extra_metadata": bundle.get("extra_metadata", {}),
    }
    return models_by_region, scalers_by_region, metadata



########### Plotting functions ###########



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

    data_start['p0'] = (data_start['pAA'] * (data_start['Initial_gen_state'] == 0) +
                       data_start['pDA'] * (data_start['Initial_gen_state'] == 1) +
                       data_start['pOA'] * (data_start['Initial_gen_state'] == 2))
    data_start['p1'] = (data_start['pAD'] * (data_start['Initial_gen_state'] == 0) +
                       data_start['pDD'] * (data_start['Initial_gen_state'] == 1) +
                       0.0 * (data_start['Initial_gen_state'] == 2))
    data_start['p2'] = (data_start['pAO'] * (data_start['Initial_gen_state'] == 0) +
                       0.0 * (data_start['Initial_gen_state'] == 1) +
                       data_start['pOO'] * (data_start['Initial_gen_state'] == 2))


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
            if (i == 1 and j == 2) or (i == 2 and j == 1):
                # skip impossible transitions D->O and O->D
                axs[i, j].set_axis_off()
                continue
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
    n_pts_per_bin: int = 100,
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
                    # if "w_stress" in bin_data.columns:
                    #     w_im = bin_data["w_stress"].to_numpy()
                    # else:
                    #     w_im = bin_data["stress_weight"].to_numpy()
                    # w_im = bin_data["w_importance"].to_numpy()
                    w_im = bin_data["region_weight"].to_numpy()*bin_data["stress_weight"].to_numpy()
                    # w_im = np.ones_like(w_im)  # disable importance weighting for log-likelihood
                    clip_tr = np.quantile(w_im, 0.95)
                    w_im = np.minimum(w_im, clip_tr)

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
                            # w_im = df_data["w_stress"].to_numpy()
                            # w_im = df_data["w_importance"].to_numpy()
                            w_im = df_data["region_weight"].to_numpy()*df_data["stress_weight"].to_numpy()
                            clip_tr = np.quantile(w_im, 0.95)
                            w_im = np.minimum(w_im, clip_tr)
                            # w_im = np.ones_like(w_im)  # disable importance weighting for calibration error

                            w = w_cp #* w_im
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
                            # n = df_data['Data_weight'].to_numpy()
                            N = df_data['Total_gen'].to_numpy()
                            n = df_data[f"f{final_state}"].to_numpy()*df_data['Total_gen'].to_numpy()
                            # w = df_data['w_stress'].to_numpy()
                            # w = df_data["w_importance"].to_numpy()
                            w = df_data["region_weight"].to_numpy()*df_data["stress_weight"].to_numpy()
                            # w = np.ones_like(n)
                            clip_tr = np.quantile(w, 0.95)
                            w = np.minimum(w, clip_tr)

                            # w = np.ones_like(n)
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







