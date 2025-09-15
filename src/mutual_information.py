import pandas as pd
import numpy as np

# Mutual information
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from npeet import entropy_estimators as ee

import inferenceModels as im
import multiprocessing as mp

import time

from typing import Iterable
from pathlib import Path
import sys



def _is_discrete_series(s: pd.Series, discrete_features: set[str]) -> bool:
    if s.name in discrete_features:
        return True
    return pd.api.types.is_integer_dtype(s) or s.nunique(dropna=False) <= max(20, int(0.01*len(s)))

def _as_cont2d(col: pd.Series) -> np.ndarray:
    return col.to_numpy(dtype=float, copy=False).reshape(-1, 1)

def _as_disc1d(col: pd.Series) -> np.ndarray:
    return col.astype(int).to_numpy(copy=False)

def compute_mutual_information_npeet(
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

def compute_mutual_information_sklearn(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    discrete_features: Iterable[str] = (),
    k: int = 3,
    use_rows: pd.Index | None = None,
    out_csv: str = "Results/mutual_information_ranking.csv",
    standardize_continuous: bool = False,
    ) -> pd.DataFrame:
    
    
    discrete_vars = [False if f not in discrete_features else True for f in feature_names]

    X_train= df.loc[:int(0.8*len(df)), feature_names].to_numpy()
    y_train= df.loc[:int(0.8*len(df)), target_col].to_numpy().ravel()

    discrete_vars = [False if f not in discrete_features else True for f in feature_names]
    data_short = df.copy()
    X_train = data_short[feature_names].to_numpy()
    y_train = data_short[target_col].to_numpy().ravel()

    mutual_information = mutual_info_classif(X_train, y_train, discrete_features=discrete_vars,
                                         n_neighbors=3,
                                         n_jobs=min(int(mp.cpu_count()/2)-2,len(feature_names)))

    mi_df = (pd.DataFrame({"feature": list(feature_names), "mi": list(mutual_information)})
             .sort_values("mi", ascending=False, ignore_index=True))

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    mi_df.to_csv(out_csv, index=False)
    print(f"Saved MI ranking to {out_csv}")
    return mi_df

def compute_mutual_information_auto(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    library = 'npeet', # 'sklearn' or 'npeet'
    discrete_features: Iterable[str] = (),
    k: int = 3,
    use_rows: pd.Index | None = None,
    out_csv: str = "Results/mutual_information_ranking.csv",
    standardize_continuous: bool = False,
    ) -> pd.DataFrame:
    if library == 'npeet':
        return compute_mutual_information_npeet(
            df=df,
            feature_names=feature_names,
            target_col=target_col,
            discrete_features=discrete_features,
            k=k,
            use_rows=use_rows,
            out_csv=out_csv,
            standardize_continuous=standardize_continuous,
        )
    elif library == 'sklearn':
        compute_mutual_information_sklearn(
            df=df,
            feature_names=feature_names,
            target_col=target_col,
            discrete_features=discrete_features,
            k=k,
            use_rows=use_rows,
            out_csv=out_csv,
            standardize_continuous=standardize_continuous,
        )
    else:
        raise ValueError(f"Unknown library '{library}'")


