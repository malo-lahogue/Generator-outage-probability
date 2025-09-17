from __future__ import annotations

# ── Standard Library ────────────────────────────────────────────────────────────
import csv
import datetime
import importlib
import itertools
import json
import math
import os
import pickle
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Robust UTC for all Python versions (3.11+: _dt.UTC, older: _dt.timezone.utc)
# try:
#     UTC = datetime.UTC            # Python 3.11+
# except AttributeError:
#     UTC = datetime.timezone.utc   # Older versions

# ── Third-Party ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split






# Set seed for reproducibility
# np.random.seed(0)
torch.manual_seed(0)







def preprocess_data(
    failure_path: str,
    event_count_path: str,
    weather_data_path: str,
    power_data_path: str,
    feature_names: List[str],
    target: str = "Frequency",
    cause_code_n_clusters: int = 1,
    randomize: bool = False,
    state_one_hot: bool = True,
    cyclic_features: List[str] = None,
    model_per_state: bool = False,
    dropNA: bool = True,
    feature_na_drop_threshold: float = 0.2,
    sort_by_date: bool = True,
    seed: Optional[int] = 42,
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
        Pre-process and align multiple daily, state-indexed datasets (failures, event counts,
        weather, and power-load), enrich them with calendar/cyclic signals, optionally cluster
        cause codes, and construct model-ready features/targets.

        The function:
        1) Loads the four CSVs and normalizes their indices to a common MultiIndex (Date, State).
        2) Selects only requested features that actually exist; drops columns with too many NaNs.
        3) Merges weather and power-load features into the event-count table (inner join on Date/State).
        4) Adds calendar features (Season/Month/DayOfWeek/DayOfYear/Holiday/Weekend) if requested.
        5) Encodes the 'State' either as one-hot columns (State_*) or as a single numeric category.
        6) Optionally encodes any listed cyclic features into sine/cosine pairs (e.g., Month → Month_sin/cos).
        7) Builds targets:
        - If cause_code_n_clusters == 1: a single target C_0 (failures or frequency).
        - If >1: clusters cause codes via k-means over feature means, then counts failures per cluster
            and produces targets C_0..C_{K-1} (raw counts or frequencies).
        In all cases, also computes a global Frequency = NumFailingUnits / NumAvailUnits (clipped to [0, 1]).
        8) Formats the supervised task:
        - target == 'Frequency': returns one row per (Date, State) with targets as rates;
            sets Data_weight = NumAvailUnits (useful for weighted losses).
        - target == 'Unit_Failure': expands to one row per available unit with one-hot targets for
            failing clusters; sets Data_weight = 1.
        9) Optionally shuffles rows deterministically.
        10) Returns a float64 DataFrame containing only [features + targets + Data_weight], plus the
            final feature and target name lists.

        INPUTS:
            - failure_path (str) : Path to the CSV with unit-level failures, including at least ['EventStartDT','State','CauseCode','UnitID'].
            - event_count_path (str) : Path to the CSV with daily state-level counts, indexed by ['EventStartDT','State'] and including ['NumAvailUnits','NumFailingUnits'].
            - weather_data_path (str) : Path to the CSV with weather features indexed by ['Date','State']; columns may include the names in feature_names.
            - power_data_path (str) : Path to the CSV with power-load features indexed by ['Date','State']; columns may include the names in feature_names.
            - feature_names (List[str]) : Candidate feature columns to include from weather/power-load (plus optional calendar/cyclic features). Nonexistent names are ignored; dropped if NaN fraction > threshold.
            - target (str) : Target mode: 'Frequency' (default, per-(Date,State) rates) or 'Unit_Failure' (per-unit expansion with one-hot cluster labels).
            - cause_code_n_clusters (int) : If 1, aggregate “any cause” as a single target C_0. If >1, cluster CauseCode into this many groups and produce C_0..C_{K-1}.
            - randomize (bool) : If True, shuffles the final rows (deterministic if seed is provided).
            - state_one_hot (bool) : If True and not model_per_state, one-hot encodes the State as State_* columns; otherwise keeps a single numeric/categorical 'State' column.
            - cyclic_features (List[str]) : Feature names to encode as sine/cosine pairs (e.g., ['Month','DayOfWeek']). Skips quietly if a name is absent; handles degenerate ranges.
            - model_per_state (bool) : If True, keeps a single 'State' column for external per-state training (no one-hot).
            - dropNA (bool) : If True, drops rows with missing values after merge and feature engineering.
            - feature_na_drop_threshold (float) : Drop any feature column whose NaN fraction exceeds this threshold (e.g., 0.2 → drop if >20% NaN).
            - seed (Optional[int]) : Random seed for deterministic shuffling and clustering initialization; if None, randomness is unconstrained.

        OUTPUTS:
            - merged_count_df (pd.DataFrame) : Final model table with columns [<features> + <target_columns> + 'Data_weight']; float64 by default. For 'Unit_Failure', rows are per unit; otherwise per (Date, State).
            - feature_names (List[str]) : Final ordered list of feature columns present in merged_count_df (after one-hot/cyclic expansion and drops).
            - target_columns (List[str]) : Names of target columns, e.g., ['C_0'] or ['C_0', ..., f'C_{K-1}'] when cause_code_n_clusters == K.
    """


    rng = np.random.default_rng(seed if seed is not None else None)

    cyclic_features = list(cyclic_features or [])
    feature_names = list(feature_names)  # defensive copy

    # ---------- Load base tables ----------
    # failure data
    failure_df = pd.read_csv(failure_path)
    event_count_df = pd.read_csv(event_count_path, index_col=["EventStartDT", "State"], parse_dates=["EventStartDT"])
    # Only rows with available units make sense
    event_count_df = event_count_df[event_count_df["NumAvailUnits"] > 0].copy()
    event_count_df["Frequency"] = event_count_df["NumFailingUnits"] / event_count_df["NumAvailUnits"]

    # --  Weather data --
    weather_df = pd.read_csv(weather_data_path, index_col=["Date", "State"], parse_dates=["Date"], usecols=lambda col: col not in ["Unnamed: 0"])
    keep_weather_features = (set(feature_names) & set(weather_df.columns)) - {"Date", "State"} # keep requested features that exist in weather_df, excluding index names
    weather_df = weather_df[list(sorted(keep_weather_features))].copy()

    # drop weather cols with too many NaNs
    na_frac = weather_df.isna().mean()
    drop_cols = na_frac[na_frac > feature_na_drop_threshold].index.tolist()
    if drop_cols:
        print(f"Dropping weather columns with >{np.around(feature_na_drop_threshold*100)}% NaN: {drop_cols}")
        weather_df.drop(columns=drop_cols, inplace=True)
        feature_names = [f for f in feature_names if f not in drop_cols]

    # -- Power Load data --
    power_load_df = pd.read_csv(power_data_path, index_col=["Date", "State"], parse_dates=["Date"], usecols=lambda col: col not in ["Unnamed: 0"])
    keep_power_features = set(feature_names) & set(power_load_df.columns)
    power_load_df = power_load_df[list(sorted(keep_power_features))].copy()

    na_frac = power_load_df.isna().mean()
    drop_cols = na_frac[na_frac > feature_na_drop_threshold].index.tolist()
    if drop_cols:
        print(f"Dropping power load columns with >{np.araound(feature_na_drop_threshold*100)}% NaN: {drop_cols}")
        power_load_df.drop(columns=drop_cols, inplace=True)
        feature_names = [f for f in feature_names if f not in drop_cols]


    # ---------- Helpers ----------
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure MultiIndex (Date, State) with normalized dtypes."""
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["Date", "State"]:
            # Try to coerce from a 2-level index with any names
            if isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2:
                date_level = df.index.get_level_values(0)
                state_level = df.index.get_level_values(1)
            else:
                raise ValueError("Expected a 2-level index for all inputs.")
        else:
            date_level = df.index.get_level_values("Date")
            state_level = df.index.get_level_values("State")

        date_level = pd.to_datetime(date_level).normalize() # Ensure that dtype is datetime64[ns] and time is 00:00:00 for everyday
        # normalize states to uppercase strings for consistent joining
        state_level = pd.Index(state_level.astype(str).str.upper(), name="State")
        idx = pd.MultiIndex.from_arrays([date_level, state_level], names=["Date", "State"])
        out = df.copy()
        out.index = idx
        return out

    def mergeData(
        dataFrames : List[pd.DataFrame],
        how        : str  = "inner",
        dropna     : bool = True,
        ) -> pd.DataFrame:
        """
        Merges a list of dataframes into a single dataframe.
        
        INPUTS:
        - dataFrames: List of dataframes to merge.
        - dropna: If True, drops rows with NaN values after merging.
        - how: Type of merge to perform (e.g., 'inner', 'outer', 'left', 'right').
        OUTPUTS:
        - Merged dataframe.
        """
        dfs = [(_normalize_index(df)) for df in dataFrames]
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, how=how)
        if dropna:
            merged = merged.dropna()
        return merged.sort_index()


    #  ---------- Merge ----------
    merged_count_df = mergeData([event_count_df, weather_df, power_load_df], how="inner", dropna=True)

    failure_idx_df = failure_df.set_index(["EventStartDT", "State"])[["CauseCode", "UnitID"]]
    # Temporarily rename level so _normalize_index treats it as Date
    failure_idx_df.index = failure_idx_df.index.set_names(["Date", "State"])
    failure_merged_df = mergeData([failure_idx_df, weather_df, power_load_df], how="inner", dropna=True).drop_duplicates()


    # ---------- State handling ----------
    if "State" not in merged_count_df.columns and "State" in merged_count_df.index.names:
        merged_count_df["State"] = merged_count_df.index.get_level_values("State")

    if state_one_hot and not model_per_state:
        merged_count_df = pd.get_dummies(merged_count_df, columns=["State"], drop_first=False, dtype=int)
        if "State" in feature_names:
            feature_names.remove("State")
        feature_names += [c for c in merged_count_df.columns if c.startswith("State_")]
    else:
        if "State" not in feature_names:
            feature_names.append("State")
        if isinstance(merged_count_df.iloc[0]["State"], str) and not model_per_state:
            cats = {s: i for i, s in enumerate(np.sort(merged_count_df["State"].unique()))}
            merged_count_df["State"] = merged_count_df["State"].map(cats)


    # ---------- Calendar features ----------
    merged_count_df["Date"]   = pd.to_datetime(merged_count_df.index.get_level_values("Date"),   errors="raise")
    failure_merged_df["Date"] = pd.to_datetime(failure_merged_df.index.get_level_values("Date"), errors="raise")

    if "Season" in feature_names:
        def get_season(ts: pd.Timestamp) -> float:
            Y = ts.year
            seasons = {
                0.0: (pd.Timestamp(f"{Y}-03-20"), pd.Timestamp(f"{Y}-06-20")),  # Spring
                1.0: (pd.Timestamp(f"{Y}-06-21"), pd.Timestamp(f"{Y}-09-22")),  # Summer
                2.0: (pd.Timestamp(f"{Y}-09-23"), pd.Timestamp(f"{Y}-12-20")),  # Autumn
                3.0: (pd.Timestamp(f"{Y}-12-21"), pd.Timestamp(f"{Y+1}-03-19")),  # Winter
            }
            for s, (start, end) in seasons.items():
                if start <= ts <= end:
                    return s
            return 3.0  # Jan–Mar before Mar 20
        merged_count_df["Season"]   = merged_count_df["Date"].apply(get_season)
        failure_merged_df["Season"] = failure_merged_df["Date"].apply(get_season)

    if "Month" in feature_names:
        merged_count_df["Month"]   = merged_count_df["Date"].dt.month
        failure_merged_df["Month"] = failure_merged_df["Date"].dt.month
    if "DayOfWeek" in feature_names:
        merged_count_df["DayOfWeek"]   = merged_count_df["Date"].dt.dayofweek
        failure_merged_df["DayOfWeek"] = failure_merged_df["Date"].dt.dayofweek
    if "DayOfYear" in feature_names:
        merged_count_df["DayOfYear"]   = merged_count_df["Date"].dt.dayofyear
        failure_merged_df["DayOfYear"] = failure_merged_df["Date"].dt.dayofyear
    if "Holiday" in feature_names:
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=merged_count_df["Date"].min(), end=merged_count_df["Date"].max())
        merged_count_df["Holiday"]   = merged_count_df["Date"].isin(holidays)
        failure_merged_df["Holiday"] = failure_merged_df["Date"].isin(holidays)
    if "Weekend" in feature_names:
        merged_count_df["Weekend"]   = merged_count_df["Date"].dt.weekday >= 5
        failure_merged_df["Weekend"] = failure_merged_df["Date"].dt.weekday >= 5


    
    # ---------- Cause-code clustering & targets ----------
    def kMeans_causeCodes(
        events_df: pd.DataFrame,
        n_clusters: int,
        features_names: Sequence[str],
        max_iter: int = 300,
        ) -> dict:
        """Cluster CauseCode by mean feature vectors."""

        use_feats = [f for f in features_names if f not in {"Date", "State"} and not f.startswith("State_")]
        tmp = events_df[["CauseCode"] + use_feats].dropna().copy()
        grouped = tmp.groupby("CauseCode")[use_feats].mean()
        if grouped.empty:
            return {}
        X = StandardScaler().fit_transform(grouped)
        km = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        return {cc: int(c) for cc, c in zip(grouped.index, labels)}


    if cause_code_n_clusters > 1:
        cc2clu = kMeans_causeCodes(failure_merged_df, cause_code_n_clusters, feature_names)
        failure_merged_df["CauseCluster"] = failure_merged_df["CauseCode"].map(cc2clu)

        # counts per (Date, State, cluster)
        counts = (
            failure_merged_df
            .groupby([failure_merged_df.index, "CauseCluster"])
            .size()
            .unstack("CauseCluster", fill_value=0)
        )
        # ensure all cluster columns exist
        for c in range(cause_code_n_clusters):
            if c not in counts.columns:
                counts[c] = 0
        counts = counts.reindex(sorted(counts.columns), axis=1)

        # align to main frame, fill missing with 0
        counts = counts.reindex(merged_count_df.index, fill_value=0)

        for c in range(cause_code_n_clusters):
            merged_count_df[f"C_{c}"] = counts[c].astype(np.int64)

        merged_count_df["NumFailingUnits"] = counts.sum(axis=1)
        if target == "Frequency":
            merged_count_df[[f"C_{c}" for c in range(cause_code_n_clusters)]] = (
                merged_count_df[[f"C_{c}" for c in range(cause_code_n_clusters)]] \
                .div(merged_count_df["NumAvailUnits"], axis=0)
                .clip(lower=0, upper=1)
            )
    else:
        # Single class: any failure
        merged_count_df["C_0"] = merged_count_df["NumFailingUnits"]
        if target == "Frequency":
            merged_count_df["C_0"] = (merged_count_df["C_0"] / merged_count_df["NumAvailUnits"]).clip(0, 1)

    target_columns = [f"C_{i}" for i in range(cause_code_n_clusters)]

    # Ensure consistent global frequency
    merged_count_df["Frequency"] = (
        merged_count_df["NumFailingUnits"] / merged_count_df["NumAvailUnits"]
    ).clip(0, 1)

    # ---------- Final NA handling ----------
    if dropNA:
        merged_count_df = merged_count_df.dropna()


    # ---------- Cyclic feature encoding ----------
    for feat in list(cyclic_features):
        if feat not in merged_count_df.columns:
            # skip quietly if not present
            continue
        series = merged_count_df[feat]
        min_val, max_val = series.min(), series.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            # degenerate: encode as 0/1 constant vectors
            merged_count_df[f"{feat}_sin"] = 0.0
            merged_count_df[f"{feat}_cos"] = 1.0
        else:
            phase = 2.0 * np.pi * (series - min_val) / (max_val - min_val)
            merged_count_df[f"{feat}_sin"] = np.sin(phase)
            merged_count_df[f"{feat}_cos"] = np.cos(phase)
        if feat in feature_names:
            feature_names.remove(feat)
        feature_names += [f"{feat}_sin", f"{feat}_cos"]
        merged_count_df.drop(columns=[feat], inplace=True, errors="ignore")


    # ---------- Unit expansion vs. frequency weighting ----------
    if target == "Unit_Failure":
        # Expand to per-unit rows with one-hot cluster targets
        avail = merged_count_df["NumAvailUnits"].to_numpy(dtype=np.int64)
        fail_mat = merged_count_df[target_columns].fillna(0).to_numpy(dtype=np.int64)
        n_rows, n_clusters = fail_mat.shape
        assert np.all(avail >= fail_mat.sum(axis=1)), "NumAvailUnits must be >= sum of failures for each (Date, State)."

        total_units = int(avail.sum())
        X_base = merged_count_df[feature_names].to_numpy()
        X_rep = np.repeat(X_base, avail, axis=0)

        Y = np.zeros((total_units, n_clusters), dtype=np.int8)
        offsets = np.concatenate(([0], np.cumsum(avail)))
        for i in range(n_rows):
            pos = offsets[i]
            for k, cnt in enumerate(fail_mat[i]):
                if cnt:
                    Y[pos:pos+cnt, k] = 1
                    pos += cnt

        merged_count_df = pd.DataFrame(X_rep, columns=feature_names)
        for k, col in enumerate(target_columns):
            merged_count_df[col] = Y[:, k]
        merged_count_df["Data_weight"] = 1.0
    elif target == "Frequency":
        merged_count_df["Data_weight"] = merged_count_df["NumAvailUnits"].astype(float)

    # ---------- Shuffle (optional) ----------
    if randomize:
        merged_count_df = merged_count_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # ---------- Sort by date (optional) ----------
    if sort_by_date:
        if 'Date' in merged_count_df.columns and 'Date' in merged_count_df.index.names:
            merged_count_df = merged_count_df.drop(columns=['Date'])   
        merged_count_df = merged_count_df.sort_values(by=["Date"] + (["State"] if "State" in merged_count_df.columns else []))
        merged_count_df = merged_count_df.reset_index(drop=True)

    # ---------- Final column selection ----------
    cols = [c for c in feature_names if c in merged_count_df.columns] + target_columns + ["Data_weight"]
    merged_count_df = merged_count_df[cols].copy()

    # Use float64 consistently (most ML libs fine with float32 if you prefer)
    return merged_count_df.astype(np.float64), feature_names, target_columns




class OptSurrogateDataset(Dataset):
    """
        Lightweight Dataset wrapper to feed ML models with (X, y, w) triples.

        Supports:
        - Tabular input: X has shape (N, D) when `feature_cols` is a flat list of column names.
        - Simple sequence input: X has shape (N, T, D_t) when `feature_cols` is a list of lists
        (one sublist per timestep). Same pattern for `target_cols`.

        INPUTS:
        - df (pd.DataFrame) : Source table containing features/targets and optional 'Data_weight'.
        - feature_cols (list) : Flat list of feature names, or list-of-lists for sequences.
        - target_cols (list) : Flat list of target names, or list-of-lists for sequences.

        OUTPUTS:
        - __len__() (int) : Number of samples.
        - __getitem__(idx) (tuple[torch.Tensor, torch.Tensor, torch.Tensor]) :
        (x, y, w) where x and y are float32 tensors, and w is a float32 scalar weight.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list, target_cols: list):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not feature_cols or not target_cols:
            raise ValueError("feature_cols and target_cols must be non-empty.")

        self.data = df.copy()

        # ---------- build X ----------
        if isinstance(feature_cols[0], (list, tuple)):
            # sequence mode
            seq_len    = len(feature_cols)          # time steps
            input_size = len(feature_cols[0])       # features/hour
            # sanity checks
            if any(len(cols_t) != input_size for cols_t in feature_cols):
                raise ValueError("All timesteps in feature_cols must have the same length.")
            for cols_t in feature_cols:
                missing = [c for c in cols_t if c not in self.data.columns]
                if missing:
                    raise KeyError(f"Missing feature columns at some timestep: {missing}")

            X = np.empty((len(self.data), seq_len, input_size), dtype=np.float32)
            for t, cols_t in enumerate(feature_cols):
                X[:, t, :] = self.data[cols_t].to_numpy(dtype=np.float32)
        else:
            missing = [c for c in feature_cols if c not in self.data.columns]
            if missing:
                raise KeyError(f"Missing feature columns: {missing}")
            X = self.data[feature_cols].to_numpy(dtype=np.float32)
        self.X = X

        # ---------- build y ----------
        if isinstance(target_cols[0], (list, tuple)):
            seq_len = len(target_cols)
            out_size = len(target_cols[0])
            if any(len(cols_t) != out_size for cols_t in target_cols):
                raise ValueError("All timesteps in target_cols must have the same length.")
            for cols_t in target_cols:
                missing = [c for c in cols_t if c not in self.data.columns]
                if missing:
                    raise KeyError(f"Missing target columns at some timestep: {missing}")

            Y = np.empty((len(self.data), seq_len, out_size), dtype=np.float32)
            for t, cols_t in enumerate(target_cols):
                Y[:, t, :] = self.data[cols_t].to_numpy(dtype=np.float32)
        else:
            missing = [c for c in target_cols if c not in self.data.columns]
            if missing:
                raise KeyError(f"Missing target columns: {missing}")
            Y = self.data[target_cols].to_numpy(dtype=np.float32)
        self.y = Y

        # ---------- optional weights ----------
        if "Data_weight" in self.data.columns:
            w = self.data["Data_weight"].to_numpy(dtype=np.float32)
        else:
            w = np.ones(len(self.data), dtype=np.float32)
        self.train_weights = w

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.as_tensor(self.X[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y[idx], dtype=torch.float32)
        w = torch.as_tensor(self.train_weights[idx], dtype=torch.float32)
        return x, y, w



class GeneratorFailureProbabilityInference:
    """
        Base class providing common data prep, standardization, plotting helpers, and
        save/load utilities for surrogate models (e.g., MLP, XGBoost).

        Typical flow:
        1) Subclass builds a model and sets `self.feature_cols` and `self.target_cols`.
        2) Call `prepare_data(...)` to split and (optionally) standardize.
        3) Call subclass `train_model(...)`.
        4) Call subclass `predict(X)`.

        INPUTS (constructor):
        - verbose (bool) : If True, print informational logs.

        OUTPUTS:
        - None : Holds `self.model`, `self.val_loss`, and dataset splits.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.model: Optional[object] = None
        self.val_loss: list[float] = []

    def prepare_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        standardize: Union[bool, list[str]] = False,
        model_per_state: bool = False,
        ) -> None:
        """
            Split data into train/val/test and (optionally) standardize columns.

            Notes:
            - Requires `self.feature_cols` and `self.target_cols` to be set by the subclass
            (e.g., in `build_model`).
            - If `standardize is True`, all feature/target columns are standardized.
            - If `standardize is a list[str]`, only those columns are standardized.

            INPUTS:
            - data (pd.DataFrame) : Full dataset containing features/targets (+ optional 'Data_weight').
            - train_ratio (float) : Fraction for training set.
            - val_ratio (float) : Fraction for validation set.
            - test_ratio (float) : Fraction for held-out test set (from the tail).
            - standardize (bool | list[str]) : Whether/which columns to standardize.
            - linear_transform_m_p (tuple[float,float]) : Placeholder affine transform for targets (not implemented).
            - model_per_state (bool) : If True, intended per-state training (not implemented here).

            OUTPUTS:
            - None : Creates `self.train_data`, `self.val_data`, `self.test_data`,
                    and, if standardization was requested, `self.scaler_feature` and `self.scaler_target`.
        """
        # ---- guards ----
        for attr in ("feature_cols", "target_cols"):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise AttributeError(
                    f"{self.__class__.__name__}.prepare_data requires '{attr}' to be set (call build_model first)."
                )

        if model_per_state:
            raise NotImplementedError(
                "model_per_state is not yet implemented in this base scaffold."
            )


        if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
            raise ValueError("train_ratio, val_ratio, and test_ratio must be non-negative.")
        if train_ratio + val_ratio + test_ratio > 1.0 + 1e-9:
            raise ValueError("train_ratio + val_ratio + test_ratio must be ≤ 1.0.")

        self.dataset_df = data.copy()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.standardize = standardize
        self.model_per_state = model_per_state

        # ---- split (tail = test) ----
        N = len(self.dataset_df)
        test_size = int(round(N * test_ratio))
        ds = self.dataset_df.iloc[: N - test_size] if test_size > 0 else self.dataset_df
        self.test_data = self.dataset_df.iloc[N - test_size :] if test_size > 0 else None

        # Now allocate train/val on the remaining ds (exact partition to avoid random_split length mismatch)
        ds_len = len(ds)
        if (train_ratio + val_ratio) <= 0:
            train_size = ds_len
            val_size = 0
        else:
            train_size = int(round(ds_len * (train_ratio / (train_ratio + val_ratio))))
            train_size = max(0, min(train_size, ds_len))
            val_size = ds_len - train_size

        # ---- standardize if requested ----
        self.scaler_feature = None
        self.scaler_target = None

        if standardize is True or isinstance(standardize, list):
            # fit scalers on pre-test data only
            self.scaler_feature = StandardScaler()
            self.scaler_target = StandardScaler()

            if standardize is True:
                stand_features = list(self.feature_cols)
                stand_targets = list(self.target_cols)
            else:
                # only those listed and present
                present = set(ds.columns)
                stand_features = [c for c in self.feature_cols if c in standardize and c in present]
                stand_targets = [c for c in self.target_cols if c in standardize and c in present]

            ds = ds.copy()
            if stand_features:
                ds.loc[:, stand_features] = self.scaler_feature.fit_transform(ds[stand_features].to_numpy())
            if stand_targets:
                ds.loc[:, stand_targets] = self.scaler_target.fit_transform(ds[stand_targets].to_numpy())

        # ---- store splits ----
        if self.__class__.__name__ == "MLP":
            # torch Dataset + exact split lengths
            ds_torch = OptSurrogateDataset(ds, feature_cols=self.feature_cols, target_cols=self.target_cols)
            self.train_data, self.val_data = random_split(
                ds_torch,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            # pandas splits (shuffle for fairness)
            ds_shuf = ds.sample(frac=1.0, random_state=42).reset_index(drop=True)
            self.train_data = ds_shuf.iloc[:train_size]
            self.val_data = ds_shuf.iloc[train_size : train_size + val_size]

    # ---------------------- Plotting ----------------------

    def plot_validation_loss(self, y_scale: str = "linear") -> None:
        """
        Plot the validation loss over epochs.

        INPUTS:
        - y_scale (str) : Matplotlib yscale, e.g. 'linear' or 'log'.

        OUTPUTS:
        - None : Shows the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.val_loss)
        ax.set_yscale(y_scale)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Over Epochs')
        plt.show()

        # ---------------------- Plotting helpers (kept & cleaned) ----------------------

    def _ensure_test_df(self, test_data: pd.DataFrame | None) -> pd.DataFrame:
        """Internal: pick a test DataFrame and drop NaNs."""
        if test_data is None:
            if getattr(self, "test_data", None) is None:
                raise ValueError("Test data not prepared. Provide test_data or call prepare_data with test_ratio>0.")
            df = self.test_data.copy()
        else:
            df = test_data.copy()
        return df.dropna()

    def _y_from_any(self, split) -> np.ndarray:
        """Internal: get y array from either a pandas DataFrame (XGB) or a torch Dataset (MLP)."""
        if hasattr(split, "_datapipes") or isinstance(split, torch.utils.data.dataset.Dataset):
            # torch Dataset: iterate once (small batches) and collect y
            loader = DataLoader(split, batch_size=4096, shuffle=False)
            ys = []
            for batch in loader:
                if len(batch) == 3:
                    _, yb, _ = batch
                else:
                    _, yb = batch
                ys.append(yb.detach().cpu().numpy())
            y = np.concatenate(ys, axis=0)
            if y.ndim > 1:
                y = y.reshape(y.shape[0], -1)
            return y.squeeze()
        else:
            # pandas split
            y = split[self.target_cols].to_numpy()
            if y.ndim > 1:
                y = y.reshape(y.shape[0], -1)
            return y.squeeze()

    def _predict_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """Internal: call self.predict on DF and return flattened predictions."""
        X_cols = list(dict.fromkeys(self.feature_cols))  # keep order, drop dups
        X = df[X_cols]
        y_pred = self.predict(X)
        if y_pred.ndim > 1:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
        return y_pred.squeeze()

    def plot_train_test_split(self) -> None:
        """
        Visualize the distribution of target values in train vs test (simple strip plot).

        INPUTS:
        - None

        OUTPUTS:
        - None
        """
        if not hasattr(self, "train_data"):
            raise ValueError("No train_data. Call prepare_data first.")

        y_train = self._y_from_any(self.train_data)
        if getattr(self, "test_data", None) is not None and len(self.test_data) > 0:
            y_test = self.test_data[self.target_cols].to_numpy().reshape(len(self.test_data), -1).squeeze()
        else:
            y_test = np.array([])

        plt.figure(figsize=(7, 4))
        x_train = np.ones_like(y_train, dtype=float)
        plt.scatter(x_train, y_train, s=6, alpha=0.5, label="Train")
        if y_test.size > 0:
            x_test = np.ones_like(y_test, dtype=float) * 2.0
            plt.scatter(x_test, y_test, s=6, alpha=0.5, label="Test")
            plt.xticks([1, 2], ["Train", "Test"])
        else:
            plt.xticks([1], ["Train"])
        plt.ylabel("Target")
        plt.title("Train/Test Target Distribution")
        plt.legend()
        plt.show()

    def plot_test_samples(
        self,
        nb_samples: int | Sequence[int] = 2,
        n_bus_max: int = 10,
        test_data: pd.DataFrame | None = None,
        BUSES: list[int] | None = None,
        hours_per_bus: int = 24,
        ) -> None:
        """
        Plot per-bus hourly curves for a handful of test samples.

        INPUTS:
        - nb_samples (int | list[int]) : Number of samples (0..N-1) or explicit indices.
        - n_bus_max (int) : Max buses to overlay.
        - test_data (pd.DataFrame | None) : If None, uses self.test_data.
        - BUSES (list[int] | None) : 1-based bus indices to show; random subset if None.
        - hours_per_bus (int) : Number of consecutive target columns per bus (default 24).

        OUTPUTS:
        - None
        """
        df = self._ensure_test_df(test_data)

        if isinstance(nb_samples, int):
            samples = np.arange(min(nb_samples, len(df)), dtype=int)
        else:
            samples = np.array(list(nb_samples), dtype=int)
        df = df.iloc[samples].reset_index(drop=True)

        # predictions
        y_pred = self._predict_from_df(df)
        y_true = df[self.target_cols].to_numpy()
        y_pred = y_pred.reshape(len(df), -1)
        y_true = y_true.reshape(len(df), -1)

        if len(self.target_cols) % hours_per_bus != 0:
            raise ValueError(
                f"len(target_cols)={len(self.target_cols)} not divisible by hours_per_bus={hours_per_bus}."
            )
        n_buses = len(self.target_cols) // hours_per_bus

        if BUSES is None:
            k = min(n_buses, n_bus_max)
            BUSES = sorted(np.random.default_rng(42).choice(np.arange(1, n_buses + 1), size=k, replace=False).tolist())
        else:
            BUSES = sorted([b for b in BUSES if 1 <= b <= n_buses])

        fig, axs = plt.subplots(len(samples), 1, figsize=(15, 5 * max(1, len(samples))))
        if len(samples) == 1:
            axs = [axs]

        # legend prologue
        for i, ax in enumerate(axs):
            # dummy handles for styles
            ax.plot([], [], linestyle='-', color='black', label='True')
            ax.plot([], [], linestyle='--', marker='x', markersize=4, color='black', label='Pred')

            x0 = np.arange(hours_per_bus)
            for b in BUSES:
                lo, hi = (b - 1) * hours_per_bus, b * hours_per_bus
                ax.plot(x0, y_true[i, lo:hi], linestyle='-')
                ax.plot(x0, y_pred[i, lo:hi], linestyle='--', marker='x', markersize=4)
                x0 = x0 + hours_per_bus  # shift next bus block to the right

            ax.set_xlim(0, hours_per_bus * len(BUSES) + 1)
            ymax = max(float(np.nanmax(y_true[i])), float(np.nanmax(y_pred[i])))
            ymin = min(float(np.nanmin(y_true[i])), float(np.nanmin(y_pred[i])))
            pad = 0.1 * (ymax - ymin + 1e-6)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xticks([])
            ax.set_xlabel("Hour blocks by bus")
            ax.set_ylabel("Target")
            ax.set_title(f"Sample {samples[i]} – per-bus hourly curves")

        # single shared legend with bus labels
        handles, labels = axs[0].get_legend_handles_labels()
        bus_labels = [f"Bus {b}" for b in BUSES]
        max_cols = 10
        ncols = min(max_cols, len(bus_labels))
        fig.legend(handles[:2], labels[:2], loc='upper left', bbox_to_anchor=(0.01, 0.98))
        fig.text(0.5, 0.02, "Buses shown: " + ", ".join(bus_labels[:50]) + ("..." if len(bus_labels) > 50 else ""), ha='center')
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.show()

    def plot_validation_loss(self, y_scale: str = 'linear') -> None:
        """
        Plot the validation loss over epochs.

        INPUTS:
        - y_scale (str) : 'linear' or 'log'.

        OUTPUTS:
        - None
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.val_loss)
        ax.set_yscale(y_scale)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Over Epochs')
        plt.show()

    def plot_test_predictions(self, test_data: pd.DataFrame | None = None) -> None:
        """
        Scatter of predictions vs true target with RMSE.

        INPUTS:
        - test_data (pd.DataFrame | None) : If None, uses self.test_data.

        OUTPUTS:
        - None
        """
        df = self._ensure_test_df(test_data)
        y_pred = self._predict_from_df(df).astype(float).ravel()
        y_true = df[self.target_cols].to_numpy(dtype=float).reshape(len(df), -1).ravel()

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        plt.figure(figsize=(6, 4))
        plt.scatter(y_true, y_pred, alpha=0.5, marker='x', s=8, label=f"RMSE = {rmse:.4f}")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        plt.plot([lo, hi], [lo, hi], linestyle='--', label='Perfect')
        plt.title('Predicted vs True')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_error(self, test_data: pd.DataFrame | None = None) -> None:
        """
        Absolute error vs true target.

        INPUTS:
        - test_data (pd.DataFrame | None) : If None, uses self.test_data.

        OUTPUTS:
        - None
        """
        df = self._ensure_test_df(test_data)
        y_pred = self._predict_from_df(df).astype(float).ravel()
        y_true = df[self.target_cols].to_numpy(dtype=float).reshape(len(df), -1).ravel()

        ae = np.abs(y_true - y_pred)
        mae = float(ae.mean())

        plt.figure(figsize=(6, 4))
        plt.scatter(y_true, ae, alpha=0.5, marker='x', s=8, label=f"MAE = {mae:.4f}")
        plt.axhline(0, linestyle='--', color='tab:red')
        plt.title('Absolute Error vs True')
        plt.xlabel('True')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_quantile_loss(
        self,
        test_data: pd.DataFrame | None = None,
        y_pred: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        relative: bool = True,
        max_y: float | None = None,
        log_y: bool = False,
        q_plot: bool = False,
        q_density_x: bool = False,
        q_spacing: float = 0.01,
        error_percentile: list[int] = [75, 95],
        ) -> None:
        """
        Quantile-wise error bands vs true value (great for heteroscedasticity).

        INPUTS:
        - test_data (pd.DataFrame | None) : If None, uses self.test_data.
        - y_pred (np.ndarray | None) : Precomputed predictions (else computed).
        - y_test (np.ndarray | None) : Precomputed truths (else from test_data).
        - relative (bool) : Plot relative (%) errors (ignoring true==0). If False, absolute errors.
        - max_y (float | None) : Optional y-max.
        - log_y (bool) : Log y-axis.
        - q_plot (bool) : Overlay data quantile curve on secondary axis.
        - q_density_x (bool) : Use equal-density spacing on x-axis.
        - q_spacing (float) : Quantile bin size (0<..≤1).
        - error_percentile (list[int]) : Extra percentiles to fill between.

        OUTPUTS:
        - None
        """
        if (y_pred is None) != (y_test is None):
            raise ValueError("If y_pred is provided, y_test must also be provided (and vice-versa).")

        if y_pred is None:
            df = self._ensure_test_df(test_data)
            y_pred = self._predict_from_df(df)
            y_test = df[self.target_cols].to_numpy()
            y_pred = y_pred.reshape(len(df), -1)
            y_test = y_test.reshape(len(df), -1)

        pred = np.asarray(y_pred, dtype=float).ravel()
        true = np.asarray(y_test, dtype=float).ravel()
        if pred.shape[0] != true.shape[0]:
            raise ValueError(f"Length mismatch: pred={pred.shape[0]} vs true={true.shape[0]}")

        abs_err = np.abs(true - pred)

        # separate zeros to avoid division-by-zero for relative errors
        mask_nz = true != 0
        true_nz = true[mask_nz]
        err_nz = abs_err[mask_nz]
        err_zeros = abs_err[~mask_nz]

        if relative:
            err_nz = (err_nz / np.abs(true_nz)) * 100.0

        # quantile bins on true_nz
        qs = np.arange(0, 1 + q_spacing, q_spacing)
        quantiles = np.quantile(true_nz, qs)
        # per-bin errors (allow empty bins -> NaN)
        binned = []
        for low, high in zip(quantiles[:-1], quantiles[1:]):
            m = (true_nz >= low) & (true_nz <= high)
            vals = err_nz[m]
            binned.append(vals if vals.size else np.array([np.nan]))

        mean_e = np.array([np.nanmean(b) for b in binned])
        min_e  = np.array([np.nanmin(b)  for b in binned])
        max_e  = np.array([np.nanmax(b)  for b in binned])
        perc_e = {p: np.array([np.nanpercentile(b, p) for b in binned]) for p in error_percentile}

        # x positions
        if q_density_x:
            x_vals = np.linspace(0, 1, len(mean_e))
        else:
            x_vals = quantiles[1:]

        fig, ax = plt.subplots(figsize=(10, 5))

        # show the (absolute) error at true==0 as reference point when plotting absolute errors
        if not relative and err_zeros.size > 0:
            ax.scatter([x_vals[0]], [np.nanmean(err_zeros)], s=24, alpha=1.0)
            ax.scatter([x_vals[0]], [np.nanpercentile(err_zeros, 75)], s=18, alpha=0.6)
            ax.scatter([x_vals[0]], [np.nanpercentile(err_zeros, 95)], s=12, alpha=0.3)

        # bands
        for p, arr in perc_e.items():
            ax.fill_between(x_vals, min_e, arr, alpha=1 - 0.85 * (p / 100) ** 2, label=f'{p}th pct error')
        ax.fill_between(x_vals, min_e, max_e, alpha=0.1, label='min/max')

        ax.plot(x_vals, mean_e, label='mean error')

        if max_y is not None:
            ax.set_ylim(0, max_y)
        else:
            finite = np.isfinite(np.concatenate([mean_e[None, :], min_e[None, :], max_e[None, :], *[v[None, :] for v in perc_e.values()]], axis=0))
            ymax = np.nanmax(np.where(finite, np.concatenate([mean_e[None, :], max_e[None, :]], axis=0), np.nan))
            if np.isfinite(ymax):
                ax.set_ylim(0, ymax * 1.05)

        if log_y:
            ax.set_yscale('log')

        if q_plot:
            ax2 = ax.twinx()
            if q_density_x:
                ax2.plot(x_vals, np.linspace(0, 100, len(x_vals)))
            else:
                ax2.plot(quantiles, np.linspace(0, 100, len(quantiles)))
            ax2.set_ylabel('Data percentile (%)')
            ax2.set_ylim(0, 100)

        if q_density_x:
            # show a few value labels for readability
            xticks_pos = np.linspace(0, 1, 5)
            xticks_lbl = [float(np.quantile(true_nz, q)) for q in xticks_pos]
            ax.set_xticks(xticks_pos)
            ax.set_xticklabels([f"{v:.2g}" for v in xticks_lbl])

        ax.set_xlabel("True value" + (" (equal-density bins)" if q_density_x else ""))
        ax.set_ylabel("Relative abs. error (%)" if relative else "Absolute error")
        ax.set_title("Quantile-wise error vs true")
        fig.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Error bands")
        plt.tight_layout()
        plt.show()

    def plot_feature_mapping(self, max_features: int = 10, test_data: pd.DataFrame | None = None) -> None:
        """
        Scatter target/prediction vs each top feature, weighted by 'Data_weight' if present.

        INPUTS:
        - max_features (int) : Maximum number of features to show (skips one-hot 'State_' by default).
        - test_data (pd.DataFrame | None) : If None, uses self.test_data.

        OUTPUTS:
        - None
        """
        df = self._ensure_test_df(test_data)
        X_test = df[self.feature_cols]
        y_true = df[self.target_cols].to_numpy()
        y_pred = self.predict(X_test)

        # weights
        if "Data_weight" in df.columns:
            data_weight = df["Data_weight"].to_numpy(dtype=float)
            data_weight = (data_weight / data_weight.max()) if data_weight.max() > 0 else np.ones_like(data_weight)
            data_weight = 0.1 + 0.9 * data_weight
        else:
            data_weight = np.ones(len(df))

        # importance
        imp_dict = self.get_feature_importance()
        imp_df = pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"]).sort_values(
            by="Importance", ascending=False
        )

        # plot
        rows = int(np.ceil(max_features / 5))
        fig, axs = plt.subplots(rows, 5, figsize=(12, 3 * rows))
        axs = axs.flatten()
        i = 0
        for _, row in imp_df.iterrows():
            feat = str(row["Feature"])
            if feat.startswith("State_"):
                continue
            if feat not in X_test.columns:
                continue
            if i >= max_features:
                break

            ax = axs[i]
            x = X_test[feat].to_numpy()
            yt = y_true.reshape(len(df), -1).mean(axis=1)  # collapse multi-target -> mean for 2D scatter
            yp = y_pred.reshape(len(df), -1).mean(axis=1)

            ax.scatter(x, yt, s=6, alpha=data_weight, label='True')
            ax.scatter(x, yp, s=6, alpha=data_weight, marker='x', label='Pred')
            ax.set_title(f"{feat} : {row['Importance']:.2g}")
            if i == 0:
                ax.legend(loc='best')
            i += 1

        # hide any unused axes
        for j in range(i, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()


class EarlyStopper:
    """
    Stop only when BOTH conditions hold:
      • No-improve: best hasn't improved by min_delta for `patience` epochs, AND
      • Flat-window: variability in last `flat_patience` epochs <= flat_delta (abs or relative).
    Guardrails:
      • Optional max_bad_epochs (hard cap).
    """
    def __init__(self,
                 min_delta: float = 0.0,
                 patience: int = 15,
                 burn_in: int = 10,
                 # flat-window
                 flat_delta: float | None = None,
                 flat_patience: int | None = None,
                 flat_mode: str = "iqr",          # "iqr" is robust
                 rel_flat: float | None = 2e-3,   # e.g., 0.2% of |best|; None to disable
                 # guardrail
                 max_bad_epochs: int | None = None):
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.burn_in = int(burn_in)
        self.flat_delta = flat_delta
        self.flat_patience = int(flat_patience or patience)
        self.flat_mode = flat_mode
        self.rel_flat = rel_flat
        self.max_bad_epochs = max_bad_epochs

        self.best = np.inf
        self.best_state = None
        self.epoch = 0
        self.epochs_since_best = 0
        self.window = deque(maxlen=self.flat_patience)
        self.stop_reason = None

    def _flat_metric(self, arr):
        if self.flat_mode == "iqr":
            q75, q25 = np.percentile(arr, [75, 25])
            return float(q75 - q25)
        return float(np.max(arr) - np.min(arr))  # range

    def step(self, val_loss: float, model=None):
        self.epoch += 1
        self.window.append(val_loss)

        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = val_loss
            self.epochs_since_best = 0
            if model is not None:
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
        else:
            self.epochs_since_best += 1

        if self.epoch < self.burn_in:
            return False

        if self.max_bad_epochs is not None and self.epochs_since_best >= self.max_bad_epochs:
            self.stop_reason = f"max_bad_epochs {self.epochs_since_best} ≥ {self.max_bad_epochs}"
            return True

        if (self.epochs_since_best >= self.patience and
            self.flat_delta is not None and
            len(self.window) == self.window.maxlen):

            thr = self.flat_delta
            if self.rel_flat is not None:
                thr = max(thr or 0.0, self.rel_flat * max(1e-12, abs(self.best)))

            fluct = self._flat_metric(list(self.window))
            if fluct <= thr:
                self.stop_reason = (f"AND stop: no-improve {self.epochs_since_best} ≥ {self.patience} "
                                    f"AND flat-window {self.flat_mode}={fluct:.3e} ≤ {thr:.3e}")
                return True
        return False


class MLP(GeneratorFailureProbabilityInference):
    """
        Multi-Layer Perceptron surrogate (PyTorch).

        Provides:
        - Flexible architecture construction.
        - Loss-aware training with optional L1/L2 regularization and sample weights.
        - Predict that honors any standardization set in `prepare_data`.
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.model = None
        self.val_loss: list[float] = []

        self.pytorch_activation_functions = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
        }
        self.pytorch_optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
        }

    def build_model(
        self,
        feature_cols: list[str],
        target_cols: list[str],
        hidden_sizes: Tuple[int, ...],
        activations: Tuple[str, ...],
        out_act_fn: Optional[str] = None,
        ) -> None:
        """
        Build the MLP computation graph.

        INPUTS:
        - feature_cols (list[str]) : Flat list of input feature names.
        - target_cols (list[str]) : Flat list of target names.
        - hidden_sizes (tuple[int,...]) : Hidden layer sizes, e.g., (256, 128, 64).
        - activations (tuple[str,...]) : Names per hidden layer (e.g., ('relu','relu','relu')).
        - out_act_fn (str | None) : Optional output activation (e.g. 'sigmoid').

        OUTPUTS:
        - None : Sets self.model and self._build_spec for checkpointing.
        """
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.hidden_sizes = hidden_sizes
        self.activations = activations

        in_dim = len(self.feature_cols)
        out_dim = len(self.target_cols)

        if len(hidden_sizes) != len(activations):
            raise ValueError("hidden_sizes and activations must have the same length.")

        model = nn.Sequential()
        last_dim = in_dim
        for l, (h, act_name) in enumerate(zip(hidden_sizes, activations)):
            if act_name not in self.pytorch_activation_functions:
                raise KeyError(f"Unknown activation '{act_name}'.")
            model.add_module(f'linear_{l}', nn.Linear(last_dim, h))
            model.add_module(f'activation_{l}', self.pytorch_activation_functions[act_name]())
            last_dim = h

        model.add_module('linear_out', nn.Linear(last_dim, out_dim))
        if out_act_fn is not None:
            if out_act_fn not in self.pytorch_activation_functions:
                raise KeyError(f"Unknown out_act_fn '{out_act_fn}'.")
            model.add_module('out_activation', self.pytorch_activation_functions[out_act_fn]())

        self.model = model

        # record rebuild spec
        self._build_spec = {
            "builder": "build_model",
            "kwargs": {
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "hidden_sizes": hidden_sizes,
                "activations": activations,
                "out_act_fn": out_act_fn,
            },
        }

        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.verbose:
            print(self.model)
            print(f"Input dim: {in_dim} | Output dim: {out_dim} | Trainable params: {self.num_parameters:,}")

    def _make_loss(self, loss_name: str, weighted: bool):
        reduction = 'none' if weighted else 'mean'
        loss_name = loss_name.lower()
        if loss_name == 'mse':
            return nn.MSELoss(reduction=reduction)
        if loss_name == 'mae':
            return nn.L1Loss(reduction=reduction)
        if loss_name == 'logloss':
            return nn.BCELoss(reduction=reduction)  # expects probs in [0,1]
        if loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss(reduction=reduction)  # expects logits + class indices
        raise ValueError(f"Unknown loss '{loss_name}'")

    def train_model(self,
                optimizer: Literal['adam','sgd','rmsprop']='adam',
                loss: Literal['mse','mae','logloss','cross_entropy']='mse',
                regularization_type='L2', lambda_reg=1e-3,
                epochs: int = 200, batch_size: int = 200, lr: float = 1e-3,
                weights_data: bool = False,
                device: str = 'cpu',
                # smart early stopping
                early_stopping: bool = True,
                patience: int = 20, min_delta: float = 0.0,
                flat_delta: float | None = None,
                flat_patience: int | None = None,
                flat_mode: str = "range",          # or "iqr"
                rel_flat: float | None = 2e-3,
                burn_in: int = 10,
                # stability & scheduling
                grad_clip_norm: Optional[float] = None,
                lr_scheduler: Optional[Literal['plateau','cosine','onecycle']] = None,
                scheduler_kwargs: Optional[dict] = None) -> None:
        """
            Train the MLP with optional data weights, smart early stopping, grad clipping,
            and LR scheduling.

            INPUTS:
            - optimizer (str): {'adam','sgd','rmsprop'}.
            - loss (str): {'mse','mae','logloss','cross_entropy'}.
            - regularization_type (str|None): 'L1', 'L2', or None.
            - lambda_reg (float|list[float]): Scalar or per-epoch strengths (len == epochs).
            - weights_data (bool): Use 'Data_weight' from Dataset in loss reduction.
            - epochs (int): Number of epochs.
            - batch_size (int): Batch size.
            - lr (float): Learning rate.
            - device (str): 'cpu' or 'cuda'.

            OUTPUTS:
            - None
        """
        # --- regularization schedule ---
        if regularization_type is not None and not (
            (isinstance(lambda_reg, (float, int)) and float(lambda_reg) > 0.0)
            or (isinstance(lambda_reg, list) and len(lambda_reg) > 0)
        ):
            raise ValueError("With regularization, lambda_reg must be > 0 (float) or a non-empty list.")

        if isinstance(lambda_reg, (float, int)):
            lambda_reg_arr = np.full(epochs, float(lambda_reg), dtype=np.float32)
        else:
            if len(lambda_reg) != epochs:
                raise ValueError("lambda_reg list must have length equal to epochs.")
            lambda_reg_arr = np.asarray(lambda_reg, dtype=np.float32)

        # --- basic setup ---
        if optimizer not in self.pytorch_optimizers:
            raise KeyError(f"Unknown optimizer '{optimizer}'.")
        self.optimizer_name = optimizer
        self.loss_fn_name = loss
        self.model.to(device)
        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(self.val_data,   batch_size=batch_size, shuffle=False)

        optim = self.pytorch_optimizers[optimizer](self.model.parameters(), lr=lr)
        loss_fn = self._make_loss(loss, weighted=weights_data)

        # --- LR scheduler (optional) ---
        scheduler = None
        scheduler_kwargs = scheduler_kwargs or {}
        if lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min',
                factor=scheduler_kwargs.get('factor', 0.5),
                patience=scheduler_kwargs.get('patience', 5),
                cooldown=scheduler_kwargs.get('cooldown', 0),
                min_lr=scheduler_kwargs.get('min_lr', 1e-6)
            )
        elif lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=max(1, epochs)
            )
        elif lr_scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim, max_lr=lr,
                steps_per_epoch=max(1, len(train_loader)), epochs=epochs
            )

        # --- weighted reduction helper ---
        def _reduce_elemwise_loss(tensor: torch.Tensor, w: Optional[torch.Tensor]):
            # tensor: [B] or [B, ...] -> mean over feature dims to per-sample, then (weighted) mean over batch
            if tensor.ndim > 1:
                per_sample = tensor.view(tensor.size(0), -1).mean(dim=1)
            else:
                per_sample = tensor
            if w is None:
                return per_sample.mean()
            denom = torch.clamp(w.sum(), min=1e-12)
            return (per_sample * w).sum() / denom

        # --- one epoch over a loader ---
        def step(loader, train: bool, epoch: int):
            self.model.train() if train else self.model.eval()
            total, n = 0.0, 0
            with torch.set_grad_enabled(train):
                for batch in loader:
                    if len(batch) == 3:
                        xb, yb, wb = batch
                        wb = wb.to(device) if weights_data else None
                    else:
                        xb, yb = batch
                        wb = None

                    xb = xb.to(device)
                    yb = yb.to(device)
                    yhat = self.model(xb)
                    elem = loss_fn(yhat, yb)                # elementwise loss
                    loss_val = _reduce_elemwise_loss(elem, wb)

                    # regularization
                    if regularization_type == 'L1':
                        l1 = sum(p.abs().sum() for p in self.model.parameters())
                        loss_val = loss_val + lambda_reg_arr[epoch-1] * l1 / self.num_parameters
                    elif regularization_type == 'L2':
                        l2 = sum(p.pow(2).sum() for p in self.model.parameters())
                        loss_val = loss_val + lambda_reg_arr[epoch-1] * l2 / self.num_parameters

                    if train:
                        optim.zero_grad()
                        loss_val.backward()
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                        optim.step()

                    bs = xb.size(0)
                    total += loss_val.item() * bs
                    n     += bs
            return total / max(1, n)

        # --- early stopper ---
        stopper = EarlyStopper(min_delta=min_delta, patience=patience, burn_in=burn_in,
                            flat_delta=flat_delta, flat_patience=flat_patience,
                            flat_mode=flat_mode, rel_flat=rel_flat)

        # --- training loop ---
        for ep in range(1, epochs + 1):
            train_loss = step(train_loader, True, ep)
            val_loss   = step(val_loader,   False, ep)
            self.val_loss.append(val_loss)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()

            if self.verbose and (ep % 10 == 0 or ep == 1):
                print(f"Epoch {ep:03d}: train={train_loss:.4e} | val={val_loss:.4e}")

            if early_stopping and stopper.step(val_loss, model=self.model):
                if self.verbose:
                    print(f"[early stop] {stopper.stop_reason} at epoch {ep}")
                break

        # --- restore best weights ---
        if early_stopping and stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)

    def reset_model(self) -> None:
        """
        Reset the model's weights (in-place) using each layer's `reset_parameters`, if available.

        INPUTS:
        - None

        OUTPUTS:
        - None
        """
        if self.model is not None:
            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            if self.verbose:
                print("Model weights have been reset.")

    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict targets for new inputs, honoring training-time standardization.

        INPUTS:
        - X (pd.DataFrame) : Inputs in original (non-standardized) scale; must contain `self.feature_cols`.

        OUTPUTS:
        - y_pred (np.ndarray) : Predictions in original scale, shape (N, T).
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        device = next(self.model.parameters()).device
        X_df = X.copy()

        # standardize selected features
        if self.standardize is True and self.scaler_feature is not None:
            X_df.loc[:, self.feature_cols] = self.scaler_feature.transform(X_df[self.feature_cols].to_numpy())
            stand_targets = list(self.target_cols)  # all will be inverse-transformed
        elif isinstance(self.standardize, list) and self.scaler_feature is not None:
            stand_feat = [c for c in self.feature_cols if c in self.standardize and c in X_df.columns]
            X_df.loc[:, stand_feat] = self.scaler_feature.transform(X_df[stand_feat].to_numpy())
            stand_targets = [c for c in self.target_cols if c in self.standardize]
        else:
            stand_targets = []

        X_np = X_df[self.feature_cols].to_numpy(dtype=np.float32)
        y_pred = self.model(torch.tensor(X_np, dtype=torch.float32, device=device)).detach().cpu().numpy()

        # inverse-transform targets if we standardized them
        if self.scaler_target is not None and stand_targets:
            y_df = pd.DataFrame(y_pred, columns=self.target_cols)
            y_df.loc[:, stand_targets] = self.scaler_target.inverse_transform(y_df[stand_targets].to_numpy())
            y_pred = y_df[self.target_cols].to_numpy(dtype=np.float32)

        return y_pred

    # ---------------------- Save / Load ----------------------

    def save_model(self, model_path: str) -> None:
        """
        Save a self-describing checkpoint that can rebuild the model and restore metadata.

        INPUTS:
        - model_path (str) : Destination path.

        OUTPUTS:
        - None : Writes a single file with tensors + metadata.
        """
        if self.model is None:
            raise ValueError("No model to save. Did you call build_model()?")

        build_spec = getattr(self, "_build_spec", None)
        if build_spec is None:
            raise RuntimeError("Subclasses must set self._build_spec in build_model().")

        def _pickle_or_none(obj):
            try:
                return pickle.dumps(obj) if obj is not None else None
            except Exception:
                return None

        checkpoint = {
            "format_version": 2,
            "saved_at": datetime.datetime.now().astimezone().isoformat(),
            "libs": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": StandardScaler.__module__.split('.')[0],
            },
            "model": {
                "module": self.__class__.__module__,
                "classname": self.__class__.__name__,
                "build_spec": build_spec,
                "state_dict": self.model.state_dict(),
            },
            "data": {
                "feature_cols": getattr(self, "feature_cols", None),
                "target_cols": getattr(self, "target_cols", None),
                "standardize": getattr(self, "standardize", False),
                "scaler_feature": _pickle_or_none(getattr(self, "scaler_feature", None)),
                "scaler_target": _pickle_or_none(getattr(self, "scaler_target", None)),
            },
            "train": {
                "optimizer_name": getattr(self, "optimizer_name", None),
                "loss_fn_name": getattr(self, "loss_fn_name", None),
                "val_loss": list(getattr(self, "val_loss", [])),
                "num_parameters": getattr(self, "num_parameters", None),
            },
        }
        torch.save(checkpoint, model_path)
        if self.verbose:
            print(f"Saved checkpoint to: {model_path}")

    @classmethod
    def load_model(cls, model_path: str, map_location: str = "cpu", verbose: bool = True):
        """
        Rebuild an MLP (or subclass) instance from a checkpoint.

        INPUTS:
        - model_path (str) : Path to saved checkpoint.
        - map_location (str) : Torch map_location for loading.
        - verbose (bool) : Print rebuild info.

        OUTPUTS:
        - obj (MLP | subclass) : Reconstructed object with weights & scalers restored.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        ckpt = torch.load(model_path, map_location=map_location, weights_only=False)

        mod_name = ckpt["model"]["module"]
        class_name = ckpt["model"]["classname"]
        build_spec = ckpt["model"]["build_spec"]
        state_dict = ckpt["model"]["state_dict"]

        module = importlib.import_module(mod_name)
        klass = getattr(module, class_name)
        obj = klass(verbose=verbose)

        data_section = ckpt.get("data", {})
        obj.feature_cols = data_section.get("feature_cols", None)
        obj.target_cols = data_section.get("target_cols", None)
        obj.standardize = data_section.get("standardize", False)

        if not build_spec or "builder" not in build_spec or "kwargs" not in build_spec:
            raise RuntimeError("Invalid or missing build_spec in checkpoint.")
        builder = getattr(obj, build_spec["builder"])
        builder(**build_spec["kwargs"])

        missing, unexpected = obj.model.load_state_dict(state_dict, strict=False)
        if verbose and (missing or unexpected):
            print(f"[load_model] Missing keys: {missing}")
            print(f"[load_model] Unexpected keys: {unexpected}")

        def _unpickle_or_none(b):
            try:
                return pickle.loads(b) if b is not None else None
            except Exception:
                return None

        obj.scaler_feature = _unpickle_or_none(data_section.get("scaler_feature", None))
        obj.scaler_target = _unpickle_or_none(data_section.get("scaler_target", None))

        train_section = ckpt.get("train", {})
        obj.optimizer_name = train_section.get("optimizer_name", None)
        obj.loss_fn_name = train_section.get("loss_fn_name", None)
        obj.val_loss = train_section.get("val_loss", [])
        obj.num_parameters = train_section.get("num_parameters", None)

        obj.model.eval()
        if verbose:
            print(f"Loaded {class_name} from {model_path}")
            print(f"Rebuilt with {build_spec['builder']}(**{list(build_spec['kwargs'].keys())})")
        return obj

    # ---------- Numpy loss & helpers for importance ----------

    def _gather_val_arrays(self, device=None, loss_name: str = "mse"):
        """
        Collect validation arrays for importance.

        INPUTS:
        - device (str | None) : Device for torch tensors (unused; kept for API parity).
        - loss_name (str) : {'mse','mae','logloss','cross_entropy'}.

        OUTPUTS:
        - (X_val, y_val, W) (tuple[np.ndarray, np.ndarray, np.ndarray]) :
          Shapes suitable for downstream loss computations.
        """
        if not hasattr(self, "val_data"):
            raise ValueError("Validation data not found. Call prepare_data() first.")

        loader = torch.utils.data.DataLoader(self.val_data, batch_size=4096, shuffle=False)
        Xs, Ys, Ws = [], [], []
        for batch in loader:
            if len(batch) == 3:
                xb, yb, wb = batch
            else:
                xb, yb = batch
                wb = torch.ones(len(xb))
            Xs.append(xb.detach().cpu())
            Ys.append(yb.detach().cpu())
            Ws.append(wb.detach().cpu())

        X = torch.cat(Xs, dim=0)
        Y = torch.cat(Ys, dim=0)
        W = torch.cat(Ws, dim=0)

        if X.ndim > 2:
            X = X.view(X.size(0), -1)

        loss_name = loss_name.lower()
        if loss_name in ("mse", "mae", "logloss"):
            if Y.ndim > 2:
                Y = Y.view(Y.size(0), -1)
            elif Y.ndim == 1:
                Y = Y.view(-1, 1)
            return X.numpy(), Y.numpy(), W.numpy()
        elif loss_name == "cross_entropy":
            if Y.ndim == 2:
                y_idx = Y.argmax(dim=1)
            elif Y.ndim == 1:
                y_idx = Y
            else:
                y_idx = Y.view(Y.size(0), -1).argmax(dim=1)
            return X.numpy(), y_idx.numpy().astype(np.int64), W.numpy()
        else:
            raise ValueError("Unknown loss_name.")

    @torch.no_grad()
    def _predict_np(self, X_np, batch_size=4096, device=None):
        self.model.eval()
        if device is None:
            device = next(self.model.parameters()).device
        preds = []
        N = X_np.shape[0]
        for i in range(0, N, batch_size):
            xb = torch.tensor(X_np[i:i + batch_size], dtype=torch.float32, device=device)
            yb = self.model(xb)
            if yb.ndim > 2:
                yb = yb.view(yb.size(0), -1)
            preds.append(yb.detach().cpu())
        return torch.cat(preds, dim=0).numpy()

    def _loss_np(self, y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None, loss: str = 'mse') -> float:
        """
        Numpy loss to mirror training.

        INPUTS:
        - y_true (np.ndarray) : True targets.
        - y_pred (np.ndarray) : Predictions (probabilities for 'logloss', logits for 'cross_entropy').
        - weights (np.ndarray | None) : Optional per-sample weights (broadcast across outputs).
        - loss (str) : {'mse','mae','logloss','cross_entropy'}.

        OUTPUTS:
        - val (float) : Scalar loss.
        """
        def reduce_elemwise(loss_elem: np.ndarray, w: Optional[np.ndarray]) -> float:
            if loss_elem.ndim > 1:
                per_sample = loss_elem.reshape(loss_elem.shape[0], -1).mean(axis=1)
            else:
                per_sample = loss_elem
            if w is None:
                return float(per_sample.mean())
            w = w.astype(np.float64)
            denom = max(1e-12, float(w.sum()))
            return float((per_sample * w).sum() / denom)

        loss = loss.lower()
        if loss == 'mse':
            return reduce_elemwise((y_true - y_pred) ** 2, weights)
        if loss == 'mae':
            return reduce_elemwise(np.abs(y_true - y_pred), weights)
        if loss == 'logloss':  # BCE
            eps = 1e-10
            p = np.clip(y_pred, eps, 1.0 - eps)
            yt = y_true
            if yt.ndim == 1:
                yt = yt.reshape(-1, 1)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            bce = -(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p))
            return reduce_elemwise(bce, weights)
        if loss == 'cross_entropy':
            # y_pred is logits (N, C)
            z = y_pred - y_pred.max(axis=1, keepdims=True)
            log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            n = y_true.shape[0]
            ll = log_probs[np.arange(n), y_true.astype(int)]
            return reduce_elemwise(-ll, weights)
        raise ValueError("loss must be 'mse', 'mae', 'logloss', or 'cross_entropy'")

    # -------- Feature importance (permutation/gradient) + plot --------

    def get_feature_importance(
        self,
        method: str = "permutation",
        n_repeats: int = 5,
        loss: Optional[str] = None,
        batch_size: int = 4096,
        top_k: int = 20,
        device: Optional[str] = None,
        normalize: bool = True,
        return_df: bool = False,
        ):
        """
        Compute feature importance for the MLP.

        INPUTS:
        - method (str) : 'permutation' or 'gradient'.
        - n_repeats (int) : Repeats for permutation importance.
        - loss (str | None) : Loss name; defaults to training loss.
        - batch_size (int) : Batch size for eval.
        - top_k (int) : Not used here (kept for API symmetry).
        - device (str | None) : Torch device.
        - normalize (bool) : Normalize importances (perm: sum=1; grad: max=1).
        - return_df (bool) : If True, return a DataFrame; else return a dict.

        OUTPUTS:
        - importance (dict[str,float] | pd.DataFrame) : Feature -> score mapping.
        """
        if device is None:
            device = next(self.model.parameters()).device
        if loss is None:
            loss = getattr(self, "loss_fn_name", "mse")

        # flat feature names
        feat_names = list(map(str, self.feature_cols))

        method = method.lower()
        loss_l = loss.lower()

        if method == "permutation":
            X_val, y_val, W = self._gather_val_arrays(device=device, loss_name=loss_l)
            y_pred = self._predict_np(X_val, batch_size=batch_size, device=device)
            base = self._loss_np(y_val, y_pred, weights=W, loss=loss_l)

            rng = np.random.default_rng(42)
            importances = np.zeros(X_val.shape[1], dtype=float)

            X_work = X_val.copy()
            for j in range(X_val.shape[1]):
                deltas = []
                for _ in range(n_repeats):
                    rng.shuffle(X_work[:, j])
                    y_perm = self._predict_np(X_work, batch_size=batch_size, device=device)
                    L = self._loss_np(y_val, y_perm, weights=W, loss=loss_l)
                    deltas.append(L - base)
                    X_work[:, j] = X_val[:, j]
                importances[j] = np.mean(deltas)

            if normalize:
                s = importances.sum()
                if s > 0:
                    importances = importances / s

            imp = {f: float(i) for f, i in zip(feat_names, importances)}
            return pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(
                "Importance", ascending=False
            ).reset_index(drop=True) if return_df else imp

        elif method == "gradient":
            if not hasattr(self, "val_data"):
                raise ValueError("Validation data not found. Call prepare_data() first.")
            loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
            self.model.eval()
            grads_sum = None
            count = 0

            for batch in loader:
                if len(batch) == 3:
                    xb, yb, _ = batch
                else:
                    xb, yb = batch
                xb = xb.to(device).float()
                if xb.ndim > 2:
                    xb = xb.view(xb.size(0), -1)
                xb.requires_grad_(True)
                yhat = self.model(xb).mean()
                self.model.zero_grad(set_to_none=True)
                yhat.backward()
                g = xb.grad.detach().abs().mean(dim=0)  # (D,)
                grads_sum = g if grads_sum is None else grads_sum + g
                count += 1

            grads_mean = (grads_sum / max(count, 1)).cpu().numpy()
            if normalize and grads_mean.max() > 0:
                grads_mean = grads_mean / grads_mean.max()

            imp = {f: float(i) for f, i in zip(feat_names, grads_mean)}
            return pd.DataFrame({"Feature": feat_names, "Importance": grads_mean}).sort_values(
                "Importance", ascending=False
            ).reset_index(drop=True) if return_df else imp

        else:
            raise ValueError("method must be 'permutation' or 'gradient'")

    def plot_feature_importance(
        self,
        method: str = "permutation",
        n_repeats: int = 5,
        loss: Optional[str] = None,
        batch_size: int = 4096,
        top_k: int = 20,
        device: Optional[str] = None,
        normalize: bool = True,
        return_df: bool = False,
        ):
        """
        Plot top-k feature importances.

        INPUTS:
        - method (str) : 'permutation' or 'gradient'.
        - n_repeats (int) : Repeats for permutation importance.
        - loss (str | None) : Loss name; defaults to training loss.
        - batch_size (int) : Batch size for eval.
        - top_k (int) : Number of top features to plot.
        - device (str | None) : Torch device.
        - normalize (bool) : Normalize importance scores.
        - return_df (bool) : Return the underlying DataFrame.

        OUTPUTS:
        - df (pd.DataFrame | None) : Top-k importance rows if requested.
        """
        imp = self.get_feature_importance(
            method=method,
            n_repeats=n_repeats,
            loss=loss,
            batch_size=batch_size,
            top_k=top_k,
            device=device,
            normalize=normalize,
            return_df=True,
        )
        k = min(top_k, len(imp))
        plt.figure(figsize=(8, max(3.5, 0.4 * k)))
        plt.barh(imp["Feature"][:k][::-1], imp["Importance"][:k][::-1])
        suffix = f" ({'norm' if normalize else 'raw'})"
        plt.xlabel(f"Importance{suffix}")
        plt.title(f"Feature Importance – {method}")
        plt.tight_layout()
        plt.show()
        return imp if return_df else None



class xgboostModel(GeneratorFailureProbabilityInference):
    """
        XGBoost surrogate wrapper (scikit-learn style).

        INPUTS (constructor):
        - verbose (bool) : Print model info.

        OUTPUTS:
        - None
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.model: Optional[xgb.XGBRegressor] = None

    def build_model(
        self,
        max_depth: int,
        eta: float,
        gamma: float,
        reg_lambda: float,
        num_boost_round: int = 100,
        feature_cols: Optional[list[str]] = None,
        target_cols: Optional[list[str]] = None,
        eval_metric: str = 'rmse',
        objective: str = 'reg:logistic',
        early_stopping_rounds: int = 10,
        subsample: float = 1.0,
        device: str = 'cpu',
        ) -> None:
        """
        Instantiate the XGBRegressor with requested hyperparameters.

        INPUTS:
        - max_depth (int) : Maximum tree depth.
        - eta (float) : Learning rate.
        - gamma (float) : Minimum loss reduction required to split.
        - reg_lambda (float) : L2 regularization term on weights.
        - num_boost_round (int) : Number of boosting rounds (n_estimators).
        - feature_cols (list[str] | None) : Input features.
        - target_cols (list[str] | None) : Target columns.
        - eval_metric (str) : e.g., 'rmse', 'logloss'.
        - objective (str) : e.g., 'reg:squarederror', 'reg:logistic'.
        - early_stopping_rounds (int) : Patience on validation metric.
        - subsample (float) : Row subsampling ratio.
        - device (str) : 'cpu' or 'cuda' (requires appropriate XGBoost build).

        OUTPUTS:
        - None
        """
        self.feature_cols = feature_cols or []
        self.target_cols = target_cols or []
        self.max_depth = max_depth
        self.eta = eta
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.num_boost_round = num_boost_round
        self.eval_metric = eval_metric
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.device = device

        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            eta=eta,
            gamma=gamma,
            reg_lambda=reg_lambda,
            n_estimators=num_boost_round,
            subsample=subsample,
            eval_metric=eval_metric,
            objective=objective,
            early_stopping_rounds=early_stopping_rounds,
            verbosity=1 if self.verbose else 0,
            device=device,
        )

        # record rebuild spec
        self._build_spec = {
            "builder": "build_model",
            "kwargs": {
                "max_depth": max_depth,
                "eta": eta,
                "gamma": gamma,
                "reg_lambda": reg_lambda,
                "num_boost_round": num_boost_round,
                "feature_cols": self.feature_cols,
                "target_cols": self.target_cols,
                "eval_metric": eval_metric,
                "objective": objective,
                "early_stopping_rounds": early_stopping_rounds,
                "subsample": subsample,
                "device": device,
            },
        }

        if self.verbose:
            print(self.model)

    def train_model(self, weights_data: bool = False) -> None:
        """
        Fit the model on train/val splits.

        INPUTS:
        - weights_data (bool) : If True, use 'Data_weight' as sample weights.

        OUTPUTS:
        - None
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        if weights_data:
            weights = self.train_data["Data_weight"].to_numpy()
        else:
            weights = np.ones(len(self.train_data), dtype=np.float32)

        X_train = self.train_data[self.feature_cols]
        y_train = self.train_data[self.target_cols]
        X_val = self.val_data[self.feature_cols]
        y_val = self.val_data[self.target_cols]

        # XGBRegressor supports early_stopping via fit(...) args
        self.model.fit(
            X_train,
            y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            # early_stopping_rounds=self.early_stopping_rounds,
        )

    def reset_model(self) -> None:
        """
        Recreate the estimator with original hyperparameters (weights discarded).

        INPUTS:
        - None

        OUTPUTS:
        - None
        """
        if self.model is not None:
            self.model = xgb.XGBRegressor(
                max_depth=self.max_depth,
                eta=self.eta,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                n_estimators=self.num_boost_round,
                subsample=self.subsample,
                eval_metric=self.eval_metric,
                objective=self.objective,
                verbosity=1 if self.verbose else 0,
                device=self.device,
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict targets for new inputs, honoring training-time standardization.

        INPUTS:
        - X (pd.DataFrame) : Inputs in original (non-standardized) scale.

        OUTPUTS:
        - y_pred (np.ndarray) : Predictions in original scale, (N, T).
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        X_df = X.copy()

        # standardize if needed
        if self.standardize is True and self.scaler_feature is not None:
            X_df.loc[:, self.feature_cols] = self.scaler_feature.transform(X_df[self.feature_cols].to_numpy())
            stand_targets = list(self.target_cols)
        elif isinstance(self.standardize, list) and self.scaler_feature is not None:
            stand_feat = [c for c in self.feature_cols if c in self.standardize and c in X_df.columns]
            X_df.loc[:, stand_feat] = self.scaler_feature.transform(X_df[stand_feat].to_numpy())
            stand_targets = [c for c in self.target_cols if c in self.standardize]
        else:
            stand_targets = []

        X_np = X_df[self.feature_cols].to_numpy(dtype=np.float32)
        y_pred = self.model.predict(X_np)

        # ensure 2D
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # inverse-transform any standardized targets
        if self.scaler_target is not None and stand_targets:
            y_df = pd.DataFrame(y_pred, columns=self.target_cols)
            y_df.loc[:, stand_targets] = self.scaler_target.inverse_transform(y_df[stand_targets].to_numpy())
            y_pred = y_df[self.target_cols].to_numpy(dtype=np.float32)

        return y_pred

    # ---------------------- Save / Load ----------------------

    def save_model(self, model_path: str) -> None:
        """
        Save a single-file checkpoint with XGBoost booster + metadata.

        INPUTS:
        - model_path (str) : Destination path.

        OUTPUTS:
        - None
        """
        if self.model is None:
            raise ValueError("No model to save. Did you call build_model()?")

        build_spec = getattr(self, "_build_spec", None)
        if build_spec is None:
            raise RuntimeError("xgboostModel must set self._build_spec in build_model().")

        def _pickle_or_none(obj):
            try:
                return pickle.dumps(obj) if obj is not None else None
            except Exception:
                return None

        # capture booster bytes if fitted
        booster_raw = None
        try:
            booster = self.model.get_booster()
            booster_raw = booster.save_raw()
        except Exception:
            booster_raw = None

        evals_result = None
        try:
            evals_result = self.model.evals_result()
        except Exception:
            pass

        checkpoint = {
            "format_version": 1,
            "saved_at": datetime.datetime.now().astimezone().isoformat(),
            "libs": {
                "xgboost": xgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": StandardScaler.__module__.split('.')[0],
            },
            "model": {
                "module": self.__class__.__module__,
                "classname": self.__class__.__name__,
                "build_spec": build_spec,
                "xgb": {
                    "sk_params": self.model.get_params(deep=False),
                    "booster_raw": booster_raw,
                    "best_iteration": getattr(self.model, "best_iteration", None),
                    "best_score": getattr(self.model, "best_score", None),
                    "evals_result": evals_result,
                },
            },
            "data": {
                "feature_cols": getattr(self, "feature_cols", None),
                "target_cols": getattr(self, "target_cols", None),
                "standardize": getattr(self, "standardize", False),
                "scaler_feature": _pickle_or_none(getattr(self, "scaler_feature", None)),
                "scaler_target": _pickle_or_none(getattr(self, "scaler_target", None)),
            },
            "train": {
                "early_stopping_rounds": getattr(self, "early_stopping_rounds", None),
            },
        }

        torch.save(checkpoint, model_path)
        if self.verbose:
            print(f"Saved XGBoost checkpoint to: {model_path}")

    @classmethod
    def load_model(cls, model_path: str, verbose: bool = True):
        """
        Reconstruct an xgboostModel instance from a checkpoint.

        INPUTS:
        - model_path (str) : Path to saved checkpoint.
        - verbose (bool) : Print rebuild info.

        OUTPUTS:
        - obj (xgboostModel) : Reconstructed object with booster and scalers restored.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        mod_name = ckpt["model"]["module"]
        class_name = ckpt["model"]["classname"]
        build_spec = ckpt["model"]["build_spec"]
        xgb_section = ckpt["model"]["xgb"]
        data_section = ckpt.get("data", {})
        train_section = ckpt.get("train", {})

        module = importlib.import_module(mod_name)
        klass = getattr(module, class_name)
        obj = klass(verbose=verbose)

        obj.feature_cols = data_section.get("feature_cols", None)
        obj.target_cols = data_section.get("target_cols", None)
        obj.standardize = data_section.get("standardize", False)

        if not build_spec or "builder" not in build_spec or "kwargs" not in build_spec:
            raise RuntimeError("Invalid or missing build_spec in checkpoint.")
        builder = getattr(obj, build_spec["builder"])
        builder(**build_spec["kwargs"])

        sk_params = xgb_section.get("sk_params", {})
        if sk_params:
            obj.model.set_params(**sk_params)

        booster_raw = xgb_section.get("booster_raw", None)
        if booster_raw is not None:
            booster = xgb.Booster()
            booster.load_model(bytearray(booster_raw))
            obj.model._Booster = booster
            try:
                obj.model.n_features_in_ = len(obj.feature_cols or [])
            except Exception:
                pass

        def _unpickle_or_none(b):
            try:
                return pickle.loads(b) if b is not None else None
            except Exception:
                return None

        obj.scaler_feature = _unpickle_or_none(data_section.get("scaler_feature", None))
        obj.scaler_target = _unpickle_or_none(data_section.get("scaler_target", None))

        obj.early_stopping_rounds = train_section.get("early_stopping_rounds", None)

        if verbose:
            print(f"Loaded {class_name} from {model_path}")
            print(f"Rebuilt with {build_spec['builder']}(**{list(build_spec['kwargs'].keys())})")
            if booster_raw is None:
                print("Note: booster was not saved (model likely not fitted yet).")
        return obj

    # ---------------------- Importance (XGBoost booster) ----------------------

    def get_feature_importance(self, importance_type: str = 'gain') -> dict[str, float]:
        """
        Return feature importance from the fitted booster.

        INPUTS:
        - importance_type (str) : One of {'weight','gain','cover'}.

        OUTPUTS:
        - importance (dict[str,float]) : Mapping feature -> importance score.
        """
        booster = self.model.get_booster()
        return booster.get_score(importance_type=importance_type)

    def plotFeatureImportance(self, importance_criterions: list[str] = ['weight', 'gain', 'cover'], n_features: int = 10) -> None:
        """
        Plot the top N features for several importance criteria.

        INPUTS:
        - importance_criterions (list[str]) : Criteria to display.
        - n_features (int) : Top features to show per criterion.

        OUTPUTS:
        - None
        """
        fig, axs = plt.subplots(1, len(importance_criterions), figsize=(6 * len(importance_criterions), n_features))
        axs = np.atleast_1d(axs)

        for i, criterion in enumerate(importance_criterions):
            importance = self.get_feature_importance(importance_type=criterion)
            imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(
                by='Importance', ascending=False
            ).reset_index(drop=True)
            axs[i].barh(imp_df['Feature'][:n_features], imp_df['Importance'][:n_features])
            axs[i].set_xlabel('Importance')
            axs[i].set_title(f'Feature Importance - {criterion}')
            axs[i].invert_yaxis()

        plt.tight_layout()
        plt.show()




# Grid search utilities

# ----------------- helpers -----------------

def _expand_grid(param_grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Turn a dict of lists into a sequence of dicts (Cartesian product).
    Example: {"a":[1,2], "b":[10]} -> {"a":1,"b":10}, {"a":2,"b":10}
    """
    keys = list(param_grid.keys())
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in (param_grid[k] for k in keys)]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def _val_loss_numpy(
    model,
    metric: str = "mse",
    weight_col: str = "Data_weight",
    use_weights: bool = True,
    ) -> float:
    """
    Compute a (possibly weighted) validation loss for any model that has:
      - model.val_data  (pd.DataFrame or torch Dataset/Subsets)
      - model.feature_cols, model.target_cols
      - model.predict(X_df) -> np.ndarray  (for DataFrame path)
    If val_data is a torch Dataset and the model implements
    `_gather_val_arrays()` and `_predict_np()`, use those (with weights).

    Supported metrics: 'mse', 'mae', 'logloss' (BCE).
    """

    def _weighted_mean(per_sample: np.ndarray, w: np.ndarray | None) -> float:
        if w is None:
            return float(per_sample.mean())
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        denom = w.sum()
        if denom <= 0:
            return float(per_sample.mean())
        return float((per_sample * w).sum() / denom)

    def _score_arrays(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None) -> float:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # Ensure 2D: (N, T)
        if y_true.ndim == 1:
            y_true = y_true[:, None]
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]

        m = metric.lower()
        if m == "mse":
            per_sample = ((y_true - y_pred) ** 2).mean(axis=1)
            return _weighted_mean(per_sample, w)
        elif m == "mae":
            per_sample = np.abs(y_true - y_pred).mean(axis=1)
            return _weighted_mean(per_sample, w)
        elif m == "logloss":
            eps = 1e-9
            p = np.clip(y_pred, eps, 1.0 - eps)
            bce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))  # (N, T)
            per_sample = bce.mean(axis=1)
            return _weighted_mean(per_sample, w)
        else:
            raise ValueError("metric must be 'mse', 'mae', or 'logloss'")

    # ---- Path 1: pandas DataFrame (e.g., XGBoost wrapper) ----
    if hasattr(model, "val_data") and isinstance(model.val_data, pd.DataFrame):
        val_df = model.val_data

        # columns
        X_cols = list(np.unique(np.array(model.feature_cols).flatten()))
        y_cols = list(np.unique(np.array(model.target_cols).flatten()))

        # data
        X_val = val_df[X_cols].copy()
        y_true = val_df[y_cols].to_numpy(dtype=np.float64)
        y_pred = model.predict(X_val)

        # weights (optional)
        w = None
        if use_weights and weight_col in val_df.columns:
            w = val_df[weight_col].to_numpy(dtype=np.float64)

        return _score_arrays(y_true, y_pred, w)

    # ---- Path 2: torch Dataset (e.g., MLP wrapper) ----
    # Prefer computing fresh weighted metric over just min(val_loss)
    has_helpers = all(
        hasattr(model, attr) for attr in ("_gather_val_arrays", "_predict_np")
    )
    if hasattr(model, "val_data") and not isinstance(model.val_data, pd.DataFrame) and has_helpers:
        # Ask the model for (X_val, y_val, weights) in numpy form
        # loss_name is only used to shape/format y in _gather_val_arrays
        loss_name = "logloss" if metric.lower() == "logloss" else metric.lower()
        X_val, y_val, W = model._gather_val_arrays(loss_name=loss_name)
        y_pred = model._predict_np(X_val)

        w = W if use_weights else None
        return _score_arrays(y_val, y_pred, w)

    # ---- Fallback: min validation loss history if present ----
    if hasattr(model, "val_loss") and len(model.val_loss) > 0:
        return float(np.min(model.val_loss))

    raise ValueError("Cannot compute validation loss: no suitable val_data or helpers found.")


def _already_done_df(csv_path: str, model_per_state) -> pd.DataFrame:
    cols = ["level", "model_name", "build_params", "train_params", "min_val_loss", "timestamp"]
    if model_per_state:
        cols += ["state"]
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # ensure expected columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)



def _row_key(level_name: str, model_name: str, build_params: Dict[str, Any], train_params: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Deterministic key for resuming."""
    return (
        str(level_name),
        str(model_name),
        json.dumps(build_params, sort_keys=True).replace(",", ";"),
        json.dumps(train_params,  sort_keys=True).replace(",", ";"),
    )



# ----------------- main search -----------------

def successive_halving_search(
    model_specs: List[Dict[str, Any]],
    data: pd.DataFrame,
    standardize: List[str] | bool,
    result_csv: str,
    train_ratio: float = 0.8,
    val_ratio: float   = 0.2,
    val_metric_per_model: Dict[str, str] | None = None,  # e.g. {"xgboostModel": "logloss", "MLP": "logloss"}
    levels: List[Dict[str, Any]] | None = None,
    top_keep_ratio: float = 0.33,
    resume: bool = True,
    subset_strategy: str = "head",   # "head" (time-respecting) or "random"
    subset_seed: int = 42,
    warm_start: bool = False,         # if True: keep MLP weights across levels (XGB still fresh)
    verbose=False,
    model_per_state: bool = False,  # if True: save model checkpoint per state (level,candidate)
    ) -> List[Tuple[int, Dict[str, Any], Dict[str, Any], float]]:
    """
        Successive halving over model families (MLP/XGBoost) and their grids.

        Returns the survivors of the last level as a list of tuples:
        (model_index, build_params, train_params, score)

        Notes on training state:
        - Default behavior (**warm_start=False**): every candidate at every level
            is trained from scratch (fresh model instance, fresh weights).
        - If **warm_start=True**: for MLP candidates only, we keep the same model
            object across levels and train **additional** epochs instead of restarting.
            XGBoost still re-trains fresh each level (continuation is non-trivial).
        INPUTS:
            - model_specs (list[dict]) : List of model specifications. Each spec:
                                                {
                                                "name": "MLP" or "xgboostModel",
                                                "constructor": callable -> returns a fresh model instance,
                                                "build_grid": dict of lists for build_model kwargs,
                                                "train_grid": dict of lists for train_model kwargs,
                                                "common_build": dict of fixed build_model kwargs,
                                                "common_train": dict of fixed train_model kwargs
                                                }
            - data (pd.DataFrame) : Full dataset with features, targets, optional 'Data_weight' column.
            - standardize (list[str] | bool) : If True, standardize all features and targets. If list, standardize only those columns.
            - result_csv (str) : Path to CSV file to log results (appended if exists).
            - train_ratio (float) : Fraction of data for training (rest for validation).
            - val_ratio (float) : Fraction of data for validation (rest for training).
            - val_metric_per_model (dict[str,str] | None) : Optional per-model validation metric overrides.
            - levels (list[dict] | None) : List of levels. Each level:
                                                {
                                                "name": str,
                                                "epochs": int,
                                                "data_cap": int | None  (max training samples; None = all)
                                                }
                                            If None, defaults to 3 levels with 40%, 80%, and 100% of data and 150, 500, 2000 epochs.
            - top_keep_ratio (float) : Fraction of candidates to keep after each level (e.g., 0.33 keeps top third).
            - resume (bool) : If True, skip candidates already present in result_csv.
            - subset_strategy (str) : "head" (time-respecting) or "random" subset when data_cap is set.
            - subset_seed (int) : Random seed for "random" subset strategy.
            - warm_start (bool) : If True, keep MLP weights across levels (XGB still fresh).
            - model_per_state (bool) : If True, save model checkpoint per state (level,candidate).
    """
    # default levels
    if levels is None:
        N = len(data)
        levels = [
            {"name": "L1-fast",   "epochs": 150,  "data_cap": int(N * 0.4)},
            {"name": "L2-medium", "epochs": 500,  "data_cap": int(N * 0.8)},
            {"name": "L3-full",   "epochs": 2000, "data_cap": None},
        ]

    if model_per_state:
        states_list = [feat_state.split('_')[1] for feat_state in data.columns if feat_state.startswith('State_')]
        print("Will search model per state")
        print(f"States found: {states_list}")

        for m in model_specs:
            m["common_build"]["feature_cols"] = [c for c in m["common_build"].get("feature_cols", []) if not c.startswith('State_')]



        if len(states_list) == 0:
            raise ValueError("model_per_state=True but no 'State_' columns found in data.")
        if len(states_list) > 1: # do the search per state
            for m in model_specs:
                m["common_build"]["feature_cols"] = [c for c in m["common_build"].get("feature_cols", []) if not c.startswith('State_')]
            for state in states_list:
                cols_one_state = [c for c in data.columns if not c.startswith('State_')]+['State_'+state]
                print(f"Searching models for state: {state}")
                data_state = data.loc[data["State_"+state]==1, cols_one_state].copy()

                successive_halving_search(
                    model_specs=model_specs,
                    data=data_state,
                    standardize=standardize,
                    result_csv=result_csv,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    val_metric_per_model=val_metric_per_model,  # e.g. {"xgboostModel": "logloss", "MLP": "logloss"}
                    levels=levels,
                    top_keep_ratio=top_keep_ratio,
                    resume=resume,
                    subset_strategy=subset_strategy,   # "head" (time-respecting) or "random"
                    subset_seed=subset_seed,
                    warm_start=warm_start,
                    verbose=False,
                    model_per_state=model_per_state,
                )
        if len(states_list) == 1: # only one state, do the search
            state_searching = states_list[0]


    # Build all (model_index, build_params, train_params)
    all_candidates: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for i, spec in enumerate(model_specs):
        build_grid = spec.get("build_grid", {}) or {}
        train_grid = spec.get("train_grid", {}) or {}
        for b in _expand_grid(build_grid):
            for t in _expand_grid(train_grid):
                all_candidates.append((i, b, t))
        # Handle the empty grid case (no build/train params)
        if not build_grid and not train_grid:
            all_candidates.append((i, {}, {}))
    # deduplicate if both branches added the empty case
    all_candidates = list({(i, json.dumps(b, sort_keys=True), json.dumps(t, sort_keys=True)) for i, b, t in all_candidates})
    all_candidates = [(i, json.loads(b), json.loads(t)) for (i, b, t) in all_candidates]
    print(f"[grid] total candidates: {len(all_candidates)}")

    # Resume logic
    done_df = _already_done_df(result_csv, model_per_state)
    done_keys = set()
    if resume and len(done_df):
        for _, r in done_df.iterrows():
            if model_per_state:
                done_keys.add((r["level"], r["model_name"], r["build_params"], r["train_params"],r["state"]))
            else:
                done_keys.add((r["level"], r["model_name"], r["build_params"], r["train_params"]))

    # Survivors structure:
    # - fresh mode: list of (mi, build_params, train_params)
    # - warm_start: list of dicts with model instance and epochs tracked for MLP
    if warm_start:
        survivors: List[Dict[str, Any]] = [
            {"mi": mi, "build_params": bp, "train_params": tp, "model": None, "trained_epochs": 0}
            for (mi, bp, tp) in all_candidates
        ]
    else:
        survivors = list(all_candidates)  # type: ignore[assignment]

    # Iterate levels
    for li, level in enumerate(levels):
        level_name  = level["name"]
        level_epochs = int(level["epochs"])
        data_cap    = level.get("data_cap", None)

        # subset data for this level
        if data_cap is not None:
            if subset_strategy == "random":
                sub_data = data.sample(n=data_cap, random_state=subset_seed).copy()
            else:
                sub_data = data.iloc[:data_cap].copy()
        else:
            sub_data = data.copy()

        scored: List[Tuple[int, Dict[str, Any], Dict[str, Any], float, Any, int]] = []

        # Evaluate each survivor
        loop_iter = survivors if warm_start else [{"mi": mi, "build_params": bp, "train_params": tp, "model": None, "trained_epochs": 0} for (mi, bp, tp) in survivors]  # unify view

        for entry in loop_iter:
            mi           = entry["mi"]
            build_params = dict(entry["build_params"])
            train_params = dict(entry["train_params"])
            existing     = entry.get("model", None)
            trained_eps  = int(entry.get("trained_epochs", 0))

            spec  = model_specs[mi]
            mname = spec["name"]

            # Merge fixed params
            build_kw = dict(spec.get("common_build", {}))
            build_kw.update(build_params)

            train_kw = dict(spec.get("common_train", {}))
            train_kw.update(train_params)

            # Cap epochs at this level
            if "epochs" in train_kw:
                train_kw["epochs"] = int(min(int(train_kw["epochs"]), level_epochs))

            # Resume skip?
            key = _row_key(level_name, mname, build_kw, train_kw)
            if model_per_state:
                key = (state_searching, *key)
            if not warm_start and resume and key in done_keys:
                prev = done_df.loc[
                    (done_df["level"] == level_name) &
                    (done_df["model_name"] == mname) &
                    (done_df["build_params"] == key[2]) &
                    (done_df["train_params"] == key[3]) &
                    (done_df["state"] == state_searching if model_per_state else True)
                ]
                if len(prev):
                    score = float(prev["min_val_loss"].iloc[0])
                    scored.append((mi, build_params, train_params, score, None, 0))
                    if verbose:
                        print(f"[resume] skip level={level_name} model={mname} -> score={score:.6g}")
                    continue

            # ---------- Build/prepare/train ----------
            model_obj = existing
            is_mlp = (mname.lower() == "mlp" or getattr(spec["constructor"](), "__class__").__name__.lower() == "mlp")

            # Build model if fresh or if not warm-starting
            if (not warm_start) or (model_obj is None) or (not is_mlp):
                model_obj = spec["constructor"]()
                model_obj.build_model(**build_kw)
                model_obj.prepare_data(
                    data=sub_data,
                    train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=1.0-(train_ratio + val_ratio),
                    standardize=standardize
                )
                # ensure fresh weights
                if hasattr(model_obj, "reset_model"):
                    model_obj.reset_model()
                model_obj.val_loss = []
                # Train for full (capped) epochs
                model_obj.train_model(**train_kw)
                trained_now = int(train_kw.get("epochs", 0))
                trained_eps = trained_now  # reset count in fresh path
            else:
                # Warm start (MLP only): keep weights; re-prepare data (in case data_cap grew)
                model_obj.prepare_data(
                    data=sub_data,
                    train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=0.0,
                    standardize=standardize
                )
                # train only the *additional* epochs at this level
                target_epochs   = int(train_kw.get("epochs", level_epochs))
                additional_eps  = max(0, target_epochs - trained_eps)
                if additional_eps > 0:
                    tmp_kw = dict(train_kw)
                    tmp_kw["epochs"] = additional_eps
                    model_obj.train_model(**tmp_kw)
                    trained_eps += additional_eps

            # ---------- Score ----------
            metric = None
            if val_metric_per_model:
                metric = (
                    val_metric_per_model.get(mname) or
                    val_metric_per_model.get(mname.lower()) or
                    val_metric_per_model.get(model_obj.__class__.__name__) or
                    val_metric_per_model.get(model_obj.__class__.__name__.lower())
                )

            if hasattr(model_obj, "val_loss") and len(model_obj.val_loss) > 0:
                score = float(np.min(model_obj.val_loss))
            else:
                score = _val_loss_numpy(model_obj, metric or "mse")

            scored.append((mi, build_params, train_params, score, model_obj if warm_start and is_mlp else None, trained_eps))

            # ---------- Log row ----------
            if result_csv:
                os.makedirs(os.path.dirname(result_csv), exist_ok=True)
                if model_per_state:
                    row = {
                        "state": state_searching,
                        "level": level_name,
                        "model_name": mname,
                        "build_params": json.dumps(build_kw, sort_keys=True).replace(",", ";"),
                        "train_params": json.dumps(train_kw, sort_keys=True).replace(",", ";"),
                        "min_val_loss": score,  # kept name for backward-compat
                        "timestamp": datetime.datetime.now().astimezone().isoformat()
                    }
                else:
                    row = {
                        "level": level_name,
                        "model_name": mname,
                        "build_params": json.dumps(build_kw, sort_keys=True).replace(",", ";"),
                        "train_params": json.dumps(train_kw, sort_keys=True).replace(",", ";"),
                        "min_val_loss": score,  # kept name for backward-compat
                        "timestamp": datetime.datetime.now().astimezone().isoformat()
                    }
                write_header = not os.path.exists(result_csv)
                with open(result_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        w.writeheader()
                    w.writerow(row)

        # ---------- Select survivors ----------
        scored.sort(key=lambda t: t[3])  # sort by score asc
        keep = max(1, int(math.ceil(len(scored) * top_keep_ratio)))

        if warm_start:
            survivors = [
                {"mi": mi, "build_params": bp, "train_params": tp, "model": mdl, "trained_epochs": te}
                for (mi, bp, tp, _, mdl, te) in scored[:keep]
            ]
        else:
            survivors = [(mi, bp, tp) for (mi, bp, tp, _, _, _) in scored[:keep]]

        # Final level returns the winners
        if li == len(levels) - 1:
            return [(mi, bp, tp, sc) for (mi, bp, tp, sc, _, _) in scored[:keep]]
