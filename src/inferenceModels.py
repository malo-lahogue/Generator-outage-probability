# Import libraries

# Data processing and manipulation
from __future__ import annotations

from typing import List, Tuple, Sequence, Optional
import numpy as np
import pandas as pd




from pandas.tseries.holiday import USFederalHolidayCalendar


from tqdm import tqdm
from collections import defaultdict

from pathlib import Path
import pickle
import os, json
from datetime import UTC, datetime
import importlib

# Visualization
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import  root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.cluster import KMeans
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from scipy.spatial import cKDTree
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

import itertools, math
from datetime import datetime as _dt







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
    randomize: bool = True,
    state_one_hot: bool = True,
    cyclic_features: List[str] = None,
    model_per_state: bool = False,
    dropNA: bool = True,
    feature_na_drop_threshold: float = 0.2,
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
        merged_count_df = merged_count_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ---------- Final column selection ----------
    cols = [c for c in feature_names if c in merged_count_df.columns] + target_columns + ["Data_weight"]
    merged_count_df = merged_count_df[cols].copy()

    # Use float64 consistently (most ML libs fine with float32 if you prefer)
    return merged_count_df.astype(np.float64), feature_names, target_columns









class OptSurrogateDataset(Dataset):
    """
    Dataset preparation for ML models.
    Holds (x, y) pairs where
      x : vector of optimization inputs / parameters
      y : vector of outputs you want the NN to predict
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_cols: list, ):
        """
        Initialize the dataset.
        INPUTS:
        - df (pd.DataFrame): The data.
        - feature_cols (list): List of features. Its shape determines the input shape. If 1 dimentional, the input is 1D, if multidimensional, the input is multidimensional (e.g for LSTMs)
        - target_cols (list): List of targets. Same comment about the shape as for features
        """
        self.data = df.copy()

        # ---------- flatten helper ----------
        def flatten(cols):
            out = []
            for c in cols:
                out.extend(flatten(c) if isinstance(c, (list, tuple)) else [c])
            return out

        # ---------- build X -------------------------------------------------
        if isinstance(feature_cols[0], (list, tuple)):
            seq_len    = len(feature_cols)          # time steps
            input_size = len(feature_cols[0])       # features/hour
            X = np.empty((len(self.data), seq_len, input_size), dtype=np.float32)
            for t, cols_t in enumerate(feature_cols):
                X[:, t, :] = self.data[cols_t].values
        else:
            X = self.data[feature_cols].values.astype(np.float32)
        self.X = X

        # ---------- build y -------------------------------------------------
        if isinstance(target_cols[0], (list, tuple)):
            seq_len    = len(target_cols)          # time steps
            out_size   = len(target_cols[0])       # targets/hour
            Y = np.empty((len(self.data), seq_len, out_size), dtype=np.float32)
            for t, cols_t in enumerate(target_cols):
                Y[:, t, :] = self.data[cols_t].values
        else:
            Y = self.data[target_cols].values.astype(np.float32)
        self.y = Y

        self.train_weights = self.data['Data_weight'].values.astype(np.float32) if 'Data_weight' in self.data.columns else np.ones(len(self.data), dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.X[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y[idx], dtype=torch.float32)
        w = torch.as_tensor(self.train_weights[idx], dtype=torch.float32)
        return x, y, w


class GeneratorFailureProbabilityInference:
    """
    Class for inferring the probability of generator failure using a machine learning model.
    1. Initialize the model with problem definition (features, targets), 
       model architecture (hidden layers, activation functions), 
       and paramters (e.g., verbosity).
    2. Build the MLP model architecture.
    3. Prepare the dataset by splitting it into training, validation, and test sets.
    4. Train the model on the training dataset.
    5. Predict the target values for new input features.
    """

    def __init__(self,
                 verbose: bool = True) -> None:
        """
        Initialize the model.
        INPUTS:
        - verbose (bool): If True, print model information.
        """
        self.verbose = verbose

        self.model = None
        self.val_loss = list()

   
    def prepare_data(self,
                    data: pd.DataFrame,
                    train_ratio: float = 0.8, val_ratio: float = 0.2, test_ratio: float = 0.0,
                    standardize: list[str] = False,
                    linear_transform_m_p=(1,0),
                    model_per_state=False)->None:
        """
        Prepares the data for machine learning models.
        Adds temporal features, as required in features_names (Season, Month, DayOfWeek, DayOfYear, Holiday, Weekend).
        Parameters:
        - data (pd.DataFrame): Data to be used for training, validation and testing.
        - train_ratio (float): Proportion of data to be used for training.
        - val_ratio (float): Proportion of data to be used for validation.
        - test_ratio (float): Proportion of data to be used for testing.
        - standardize (list[str] or bool): If True, standardize all features and targets. If a list of column names, standardize only those columns. If False, do not standardize.
        - state_one_hot (bool): If True, applies one-hot encoding to the 'State' column.
        - randomize (bool): If True, shuffles the data before splitting.
        - model_per_state (bool): If True, trains a separate model for each state.
        - dropNA (bool): If True, drops rows with NaN values.
        """

        self.dataSet = data.copy()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.standardize = standardize
        self.linear_transform_m_p = linear_transform_m_p
        self.model_per_state = model_per_state

        print("linear_transform_m_p is not yet implemented")

        if model_per_state:
            raise NotImplementedError("Model per state is not implemented yet. Please set model_per_state to False.")
        
        # self.plot_transform()


        # split the data into train, validation and test sets and standardize it if required
        N = len(data)
        train_size = int(N * train_ratio)
        val_size   = int(N * val_ratio)
        test_size  = N - train_size - val_size

        self.test_data = data.iloc[-test_size:]
        ds = data.iloc[:N - test_size] if test_size > 0 else data# self.data

        if standardize == True:
            self.scaler_target = StandardScaler()
            self.scaler_feature = StandardScaler()
            ds.loc[:,self.target_cols] = self.scaler_target.fit_transform(ds[self.target_cols])
            ds.loc[:,self.feature_cols] = self.scaler_feature.fit_transform(ds[self.feature_cols])

        elif isinstance(standardize, list):
            self.scaler_target = StandardScaler()
            self.scaler_feature = StandardScaler()
            stand_features = list(np.array(self.feature_cols)[np.isin(self.feature_cols, standardize)])
            stand_targets = list(np.array(self.target_cols)[np.isin(self.target_cols, standardize)])
            if len(stand_features) > 0:
                ds = ds.copy()
                ds.loc[:,stand_features] = self.scaler_feature.fit_transform(ds[stand_features].values)
            if len(stand_targets) > 0:
                ds = ds.copy()
                ds.loc[:,stand_targets] = self.scaler_target.fit_transform(ds[stand_targets].values)

        if self.__class__.__name__ == "MLP":
            ds = OptSurrogateDataset(ds, target_cols=self.target_cols, feature_cols=self.feature_cols)

            self.train_data, self.val_data = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        else:
            ds = ds.sample(frac=1, random_state=42).reset_index(drop=True)
            self.train_data = ds.iloc[:train_size]
            self.val_data   = ds.iloc[train_size:train_size + val_size]




        
     # ---- Plotting ----
    
    # Plotting methods

    def plot_transform(self):
        """
        Plots the affine transformation applied to the target variable.
        """
        print(f"Affine transformation of the target variable : x' = {self.linear_transform_m_p[0]}x + {self.linear_transform_m_p[1]}")
        m, p = self.linear_transform_m_p
        x = np.linspace(0, 1, 100)
        y = m * x + p

        plt.figure(figsize=(3,2))
        plt.plot(x, y, label=f"x' = {m:.3f}x + {p}", color='blue')
        plt.title('Affine Transformation of Target Variable', fontsize=10)
        plt.xlabel('Original', fontsize=10)
        plt.ylabel('Transformed', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(fontsize=10)
        plt.show()

    def plot_train_test_split(self):
        """
        Plots the training and testing data points.
        """
        plt.figure(figsize=(6, 4))
        plt.scatter(np.ones_like(self.y_train), self.y_train, color='blue', label='Train')
        plt.scatter(np.ones_like(self.y_test)*2, self.y_test, color='orange', label='Test')
        plt.title('Train/Test Split', fontsize=14)
        plt.ylabel('Target', fontsize=12)
        plt.xticks([])
        plt.legend(fontsize=12)
        plt.show()

     # Plotting functions
    
    def plot_test_samples(self, nb_samples: int = 2, n_bus_max=10, test_data=None, BUSES:list[int]=None) -> None:
        """
        Plot the predictions for a few test samples.
        """
        if test_data is None:
            if self.test_data is not None:
                test_data = self.test_data.copy()
            else:
                raise ValueError("Test data not prepared. Please input test data first.")
        if isinstance(nb_samples, int):
            samples = np.arange(nb_samples, dtype=int)
        else:
            samples = nb_samples  # Assume it's an iterable of indices
        
        test_data = test_data.iloc[samples]

        # ---------- build X_test and predict ----------------
        test_data.reset_index(drop=True, inplace=True)
        X_cols = list(np.unique(np.array(self.feature_cols).flatten()))
        X_test = test_data[X_cols]
        y_pred = self.predict(X_test)

        y_pred = y_pred.reshape(-1, len(np.array(self.target_cols).flatten()))  # Ensure y_pred is 2D

        # ---------- build y_test -------------------------------------------------
        if isinstance(self.target_cols[0], (list, tuple)):
            seq_len    = len(self.target_cols)          # time steps
            out_size   = len(self.target_cols[0])       # targets/hour
            y_test = np.empty((len(test_data), seq_len, out_size), dtype=np.float32)
            for t, cols_t in enumerate(self.target_cols):
                y_test[:, t, :] = test_data[cols_t].values
        else:
            y_test = test_data[self.target_cols].values.astype(np.float32)
        y_test = test_data.loc[samples, self.target_cols].to_numpy()
        y_test = y_test.reshape(-1, len(np.array(self.target_cols).flatten()))  # Ensure y_test is 2D

        # ---------- parameters for plotting ----------------

        n_buses = int(len(self.target_cols)/24)
        if BUSES is None:
            BUSES = np.random.choice(range(1, n_buses + 1), size=min(n_buses, n_bus_max), replace=False)
        BUSES = sorted(BUSES)  # Sort the bus numbers for consistent plotting
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']*100
        colors = colors[:len(BUSES)]
        

        # ---------- plot the predictions ----------------

        fig, axs = plt.subplots(nb_samples, 1, figsize=(15, 5*nb_samples))
        for i in range(nb_samples):
            ax = axs[i] if nb_samples > 1 else axs
            if i == 0:
                ax.plot([-2,-1], [0,0], linestyle='-', color='black', label='True values')
                ax.plot([-2,-1], [0,0], linestyle='--', marker='x', markersize=4, color='black', label='Predicted values')
                ax.legend()
            x = np.arange(24)
            for b,bus in enumerate(BUSES):
                y_t = y_test[i, (bus-1)*24:bus*24]
                y_p = y_pred[i, (bus-1)*24:bus*24]
                ax.plot(x, y_t, label=f'Bus {bus}', linestyle='-', color=colors[b])
                ax.plot(x, y_p, linestyle='--', marker='x', markersize=4, color=colors[b])
                x += 24  # Move to the next bus
            ax.set_xticks([])
            ax.set_xlim(0, 24*len(BUSES)+1)
            ax.set_ylim(min(-1, -0.1*max(np.max(y_test), np.max(y_pred))), 1.1 * max(np.max(y_test), np.max(y_pred)))
            ax.set_xlabel('Hour of the day and bus (color)')
            ax.set_ylabel('Load shedding (MW)')
            ax.set_title(f'Load Shedding Prediction - Sample {i+1}')

            handles, labels = ax.get_legend_handles_labels()
            handles = handles[2:] # remove the labels indicating true vs predicted values
            labels = labels[2:]

            # choose an upper‑bound on columns so the legend does not grow too tall
            max_cols = 10
            ncols    = min(max_cols, len(labels))

            fig.legend(handles, labels,
                    loc='upper center',       # horizontally centred
                    bbox_to_anchor=(0.5, 0),  # 5 % of fig‑height below the axes
                    ncol=ncols,               # spread across ncols columns
                    frameon=True,            # optional aesthetics
                    #fontsize='small'
                    )         # optional, keeps text tidy

    def plot_validation_loss(self, y_scale='linear') -> None:
        """
        Plot the validation loss over epochs.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.val_loss, color='blue')
        ax.set_yscale(y_scale)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Over Epochs')
        plt.show()

    def plot_quantile_loss(self, 
                           test_data=None, y_pred=None, y_test=None,
                           relative=True, max_y=None, 
                           log_y=False, 
                           q_plot=False, q_density_x=False,
                           q_spacing=0.01,
                           error_percentile = [75, 95]) -> None:
        """
        Plot the quantile loss for different quantiles.
        INPUTS:
        - test_data (pd.DataFrame): DataFrame containing the test data. If None, it will use the test data prepared during model training.
        - relative (bool): If True, plot relative errors as percentages. If False, plot absolute errors
        - max_y (float): Maximum value for the y-axis.
        - log_y (bool): If True, use a logarithmic scale for the y-axis.
        - q_plot (bool): If True, plot the quantiles.
        - q_density_x (bool): If True, scales the x-axis to have equal quantile representation
        - q_spacing (float): Spacing between quantiles. Default is 0.01.
        - error_percentile (list): List of percentiles to compute for the errors. Default is [75, 95]. We also add maximum error (100).
        - scaler (float or dict[str:float]): Value to scale the predictions and targets before computing the loss. If a dictionary is provided, it should map column types to scaling factors. (load shedding, power produced ...)
        - units (str or dict[str:float]): Units for the y-axis label. If a dictionary is provided, it should map column types to units. (load shedding, power produced ...)
        - y_pred (np.ndarray): Optional pre-computed predictions. If None, predictions will be computed from the test data.
        - y_test (np.ndarray): Optional pre-computed true values. If None, true values will be computed from the test data.
        """
        if not y_pred is None and y_test is None:
            raise ValueError("If y_pred is provided, y_test must also be provided.")
        

        # Get prediction and true values if not provided
        if y_pred is None:
            if test_data is None:
                if self.test_data is not None:
                    test_data = self.test_data.copy()
                else:
                    raise ValueError("Test data not prepared. Please input test data first.")
            else:
                test_data = test_data.copy()
                test_data.dropna(inplace=True)

            X_cols = list(np.unique(np.array(self.feature_cols).flatten()))
            X_test = test_data[X_cols] #only pass the feature columns
            y_pred = self.predict(X_test)
            y_pred = y_pred

            # ---------- build y_test -------------------------------------------------
            if isinstance(self.target_cols[0], (list, tuple)):
                seq_len    = len(self.target_cols)          # time steps
                out_size   = len(self.target_cols[0])       # targets/hour
                y_test = np.empty((len(test_data), seq_len, out_size), dtype=np.float32)
                for t, cols_t in enumerate(self.target_cols):
                    y_test[:, t, :] = test_data[cols_t].values
            else:
                y_test = test_data[self.target_cols].values.astype(np.float32)

            y_test = y_test


        pred = y_pred.flatten()
        test = y_test.flatten()







        if len(test) != len(pred):
            raise ValueError(f"Length mismatch: {len(test)} test values vs {len(pred)} predicted values.")


        # ---------- compute errors and quantiles ----------------

        errors = np.abs(test - pred)

        errors_0 = errors[test == 0]

        test_nonzero = test[test != 0]
        errors_nonzero = errors[test != 0]

        if relative:
            errors_relative = [e / y * 100 for e, y in zip(errors_nonzero, test_nonzero)]
            errors_nonzero = np.array(errors_relative)

        quantiles = np.quantile(test_nonzero, np.arange(0, 1+q_spacing, q_spacing))
        binned_errors = [errors_nonzero[(test_nonzero >= low) & (test_nonzero <= high)].tolist() for low, high in zip(quantiles[:-1], quantiles[1:])]

        mean_errors = [np.mean(bin) for bin in binned_errors]
        min_errors = [np.min(bin) for bin in binned_errors]
        max_errors = [np.max(bin) for bin in binned_errors]
        errors_percentile = {p: [np.percentile(bin, p) for bin in binned_errors] for p in error_percentile}

        # --- choose x values ---
        if q_density_x:
            # Create equally spaced positions for each quantile bin
            x_vals = np.linspace(0, 1, len(mean_errors))  # quantile positions from 0 to 1
        else:
            # Use actual quantile value ranges
            x_vals = quantiles[1:]

        fig, ax = plt.subplots(figsize=(10, 5))
        if not relative:
            ax.scatter([x_vals[0]], [np.mean(errors_0)], s=20, color='blue', alpha=1)
            ax.scatter([x_vals[0]], [np.max(errors_0)], s=20, color='blue', alpha=0.1)
            ax.scatter([x_vals[0]], [np.percentile(errors_0, 75)], s=20, color='blue', alpha=0.5)
            ax.scatter([x_vals[0]], [np.percentile(errors_0, 95)], s=20, color='blue', alpha=0.3)

        # ax.fill_between(x_vals, min_errors, errors_95, color='blue', alpha=0.3, label='95th Percentile Error')
        # ax.fill_between(x_vals, min_errors, errors_75, color='blue', alpha=0.5, label='75th Percentile Error')
        for p, err in errors_percentile.items():
            ax.fill_between(x_vals, min_errors, err, color="blue", alpha=1 - 0.85 * (p / 100)**2, label=f'{p}th Percentile Error')
        ax.fill_between(x_vals, min_errors, max_errors, color='blue', alpha=0.1, label='Min/Max Error')

        ax.plot(x_vals, mean_errors, label='Mean Error', color='darkmagenta')
        if max_y is not None:
            ax.set_ylim(0, max_y)
        else:
            ax.set_ylim(0, np.max(np.concatenate((mean_errors, max_errors, min_errors, *[errors_percentile[p] for p in error_percentile])))) 
        if log_y:
            ax.set_yscale('log')
        if q_plot:
            ax2 = ax.twinx()
            # ax2.plot(quantiles, np.linspace(0, 100, len(quantiles)), linestyle='-', color='orange', label='Quantiles')
            if q_density_x:
                ax2.plot(x_vals, np.linspace(0, 100, len(quantiles[1:])), linestyle='-', color='orange', label='Quantiles')
            else:
                ax2.plot(quantiles, np.linspace(0, 100, len(quantiles)), linestyle='-', color='orange', label='Quantiles')
            
            ax2.set_ylabel('Data Percentile (%)')
            ax2.set_ylim(0, 100)

        # ax.set_xticklabels(quantiles[1:])

        if q_density_x:
            xticks_pos = np.linspace(0, 1, 5)
            # Round down to the same first significant figure
            def round_down_sig(x):
                if x == 0:
                    return 0
                exp = int(np.floor(np.log10(abs(x))))
                factor = 10 ** exp
                return np.floor(x / factor) * factor
            
            xticks_labels = [round_down_sig(q) for q in np.quantile(test_nonzero, xticks_pos)]
            small_quantiles = np.quantile(test_nonzero, np.arange(0,1+1e-4, 1e-4))
            xticks_pos = np.interp(xticks_labels, small_quantiles, np.linspace(0, 1, len(small_quantiles)))

            ax.set_xticks(xticks_pos)
            ax.set_xticklabels(xticks_labels)


        ax.set_xlabel("True value")

        ax.set_ylabel(f"Absolute error" if not relative else 'Relative absolute error (%)')
        ax.set_title(f"Quantile Loss  vs True Values")
        fig.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Per True Value Error Statistics")
        plt.show()

    def plot_test_predictions(self, test_data=None):
        """
        Plots the predictions of the model on the test set against the true values.
        """

        if test_data is None:
            if self.test_data is not None:
                test_data = self.test_data.copy()
            else:
                raise ValueError("Test data not prepared. Please input test data first.")
        else:
            test_data = test_data.copy()
            test_data.dropna(inplace=True)

        X_cols = list(np.unique(np.array(self.feature_cols).flatten()))
        X_test = test_data[X_cols] #only pass the feature columns
        y_pred = self.predict(X_test)
        y_pred = y_pred.flatten() 

        if isinstance(self.target_cols[0], (list, tuple)):
            seq_len    = len(self.target_cols)          # time steps
            out_size   = len(self.target_cols[0])       # targets/hour
            y_test = np.empty((len(test_data), seq_len, out_size), dtype=np.float32)
            for t, cols_t in enumerate(self.target_cols):
                y_test[:, t, :] = test_data[cols_t].values
        else:
            y_test = test_data[self.target_cols].values.astype(np.float32)

        y_test = y_test.flatten()

        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.5, marker='x', s=5, color='blue', label=f"rmse = {rmse:.4f}")
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfect Prediction')
        plt.title('Model Predictions vs True Values', fontsize=14)
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend(fontsize=12)
        plt.show()

    def plot_error(self, test_data=None):
        """
        Plots the error of the model predictions on the test set.
        """
        
        if test_data is None:
            if self.test_data is not None:
                test_data = self.test_data.copy()
            else:
                raise ValueError("Test data not prepared. Please input test data first.")
        else:
            test_data = test_data.copy()
            test_data.dropna(inplace=True)

        X_cols = list(np.unique(np.array(self.feature_cols).flatten()))
        X_test = test_data[X_cols] #only pass the feature columns
        y_pred = self.predict(X_test)
        y_pred = y_pred.flatten() 

        if isinstance(self.target_cols[0], (list, tuple)):
            seq_len    = len(self.target_cols)          # time steps
            out_size   = len(self.target_cols[0])       # targets/hour
            y_test = np.empty((len(test_data), seq_len, out_size), dtype=np.float32)
            for t, cols_t in enumerate(self.target_cols):
                y_test[:, t, :] = test_data[cols_t].values
        else:
            y_test = test_data[self.target_cols].values.astype(np.float32)

        y_test = y_test.flatten()

        ae = np.abs(y_test - y_pred)
        # mape = np.abs((y_test - y_pred) / y_test) * 100

        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, ae, alpha=0.5, marker='x', s=5, color='blue', label=f"Mean Absolute Error = {np.mean(ae):.4f}")
        # plt.scatter(y_test, mape, alpha=0.5, marker='o', s=5, color='green', label=f"MAPE = {np.mean(mape):.4f}%")
        plt.axhline(0, color='red', linestyle='--', label='Zero Error')
        plt.title('Prediction Error vs True Values', fontsize=14)
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Prediction Absolute Error', fontsize=12)
        plt.legend(fontsize=12)
        plt.show()

    def plot_feature_mapping(self, max_features: int = 10, test_data=None):
        """
        Plots the mapping of features to predictions.
        INPUTS:
        - max_features (int): Maximum number of features to plot.
        - test_data (pd.DataFrame): DataFrame containing the test data. If None, it will use the test data prepared during model training.
        """
        # X_train, X_test, y_train, y_test, y_freq_test, ypred = data
        X_test = self.test_data[self.feature_cols] if test_data is None else test_data[self.feature_cols]
        y_test = self.test_data[self.target_cols] if test_data is None else test_data[self.target_cols]
        ypred = self.predict(X_test)
        data_weight = self.test_data['Data_weight'] if test_data is None else test_data['Data_weight']
        data_weight /= data_weight.max()  # Normalize weights to sum to 1
        data_weight = 0.1 +0.9* data_weight  # Scale to [0.1, 1] for better visibility

        fig, axs = plt.subplots(int(np.ceil(max_features / 5)), 5, figsize=(12, 3*int(np.ceil(max_features / 5))))

        # self.get_feature_importance()
        importance_df = pd.DataFrame(list(self.get_feature_importance().items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        # importance_df = pd.DataFrame(list(model.get_score(importance_type='gain').items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False).reset_index(drop=True)

        i = 0
        for _,row in importance_df.iterrows():
            feat = row['Feature']
            if feat.startswith('State_'):
                continue
            if i >= max_features:
                break
            # if feat in merged_count_df['State'].unique():
            #     continue
            imp = np.round(row['Importance'], 2)
            ax = axs.flatten()[i]
            var = X_test[feat].to_numpy()
            freq_h = y_test.to_numpy()
            prob = ypred
            ax.scatter(var, freq_h, marker='o', s=1, alpha=data_weight, label='Historical')
            ax.scatter(var, prob, marker='x', s=1, alpha=data_weight, label='Predicted')
            ax.set_title(f'{feat} : {imp}')
            if i == 0:
                ax.legend(loc='upper right')
            i += 1

        plt.tight_layout()


class MLP(GeneratorFailureProbabilityInference):
    """
    Class for inferring the probability of generator failure with a Multi Layer Perceptron (MLP) model using Pytorch
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)

        self.model = None
        self.val_loss = list()

        self.pytorch_activation_functions = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh
        }

        self.pytorch_optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }

    def build_model(self,
                    feature_cols: list[str], target_cols: list[str],
                    hidden_sizes: tuple, activations: tuple,
                    out_act_fn=None,
                    ) -> None:
        """
        Build the MLP model architecture.
        INPUTS:
        - feature_cols (list[str]): List of feature column names.
        - target_cols (list[str]): List of target column names.
        - hidden_sizes (tuple): Tuple of integers representing the number of neurons in each hidden layer
        - activations (tuple): Tuple of activation function names for each hidden layer.
        """
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.hidden_sizes = hidden_sizes
        self.activations = activations
        in_dim = len(self.feature_cols)
        out_dim = len(self.target_cols)
    

        last_dim = in_dim
        model = nn.Sequential()
        for l in range(len(self.hidden_sizes)):
            hidden_dim = self.hidden_sizes[l]

            act_fn = self.pytorch_activation_functions[self.activations[l]]()

            model.add_module(f'linear_{l}', nn.Linear(last_dim, hidden_dim))
            model.add_module(f'activation_{l}', act_fn)

            last_dim = hidden_dim
        model.add_module(f'linear_LAST', nn.Linear(last_dim, out_dim))
        if out_act_fn is not None:
            model.add_module('out_activation', self.pytorch_activation_functions[out_act_fn]())
        self.model = model

        # record how to rebuild this architecture
        self._build_spec = {
            "builder": "build_model",
            "kwargs": {
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "hidden_sizes": hidden_sizes,
                "activations": activations,
                "out_act_fn": out_act_fn, 
            }
        }

        self.num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.verbose:
            print(f"Model architecture: {self.model}")
            print(f"Input dimension: {in_dim}, Output dimension: {out_dim}")
            print(f"Number of parameters: {self.num_param}")

    def train_model(self,
                    optimizer: str = 'adam',  loss: str = 'mse',
                    regularization_type='L2', lambda_reg=1e-3,
                    weights_data : bool = False,
                    epochs: int = 200, batch_size: int = 200, lr: float = 1e-3,
                    device: str = 'cpu') -> None:
        """
        Train the model on the training dataset.
        INPUTS:
        - optimizer (str): The optimizer to use. Options: 'adam', 'sgd', 'rmsprop'.
        - loss_fn (str): The loss function to use. Options: 'mse', 'mae', 'logloss', 'cross_entropy'.
        - regularization_type (str): Type of regularization to apply. Options: 'L1', 'L2', or None.
        - lambda_reg (float): Regularization strength.
        - weights_data (bool): If True, use weights for the loss function.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate.
        - device (str): Device to train on. Options: 'cpu', 'cuda'.
        """
        if regularization_type is not None and not lambda_reg > 0:
            raise ValueError(f"You try to regularize with {regularization_type}. Then lambda_reg must be > 0. (currently input {lambda_reg})")
        
        if isinstance(lambda_reg, float):
            lambda_reg = np.ones(epochs) * lambda_reg
        elif isinstance(lambda_reg, list) and len(lambda_reg) != epochs:
            raise ValueError(f"If lambda_reg is a list, it must have the same length as epochs. (currently input of length {len(lambda_reg)} with {epochs} epochs)")
        
        self.optimizer_name = optimizer
        self.loss_fn_name = loss
        self.num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.model.to(device)

        train_loader = DataLoader(self.train_data, batch_size, shuffle=True)
        val_loader   = DataLoader(self.val_data,   batch_size, shuffle=False)


        optimizer = self.pytorch_optimizers[self.optimizer_name](self.model.parameters(), lr=lr)
        loss_fn = self._make_loss(self.loss_fn_name, weighted=weights_data)

        # --- training helpers --------
        def _reduce_elemwise_loss(loss_elem, w=None):
            """
            loss_elem shape:
            - Regression (MSE/MAE): [B, D] or [B, T, D]
            - BCE: same as regression
            - CE: [B] (already per-sample)
            Returns scalar loss (weighted or unweighted mean over batch).
            """
            if loss_elem.dim() == 1:
                per_sample = loss_elem  # [B]
            else:
                # average across all target dimensions per sample
                per_sample = loss_elem.view(loss_elem.size(0), -1).mean(dim=1)  # [B]

            if w is None:
                return per_sample.mean()
            # weighted mean over batch; avoid div-by-zero
            denom = w.sum().clamp_min(1e-12)
            return (per_sample * w).sum() / denom
        
        def step(loader, train: bool, epoch:int):
            if train: self.model.train()
            else:     self.model.eval()
            total, n = 0.0, 0
            with torch.set_grad_enabled(train):
                for batch in loader:
                    # Support both (x,y) and (x,y,w) just in case
                    if len(batch) == 3:
                        xb, yb, wb = batch
                        wb = wb.to(device)
                        if not weights_data:
                            wb = None
                    else:
                        xb, yb = batch
                        wb = None

                    xb, yb = xb.to(device), yb.to(device)
                    pred = self.model(xb)

                    loss_elem = loss_fn(pred, yb)  # elementwise loss
                    loss = _reduce_elemwise_loss(loss_elem, w=wb)

                    # regularization
                    if regularization_type == 'L1':
                        l1 = sum(p.abs().sum() for p in self.model.parameters())
                        loss = loss + lambda_reg[epoch-1] * l1 / self.num_parameters
                    elif regularization_type == 'L2':
                        l2 = sum(p.pow(2).sum() for p in self.model.parameters())
                        loss = loss + lambda_reg[epoch-1] * l2 / self.num_parameters

                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    bs = xb.size(0)
                    total += loss.item() * bs
                    n     += bs
            return total / max(n, 1)
            

        # --- training loop ----------
        for ep in range(1, epochs + 1):
            train_loss = step(train_loader, True, ep)
            val_loss   = step(val_loader,   False, ep)
            self.val_loss.append(val_loss)
            if (self.verbose) and (ep % 10 == 0 or ep == 1):
                print(f"Epoch {ep:03d}: train={train_loss:.4e} | val={val_loss:.4e}")

    def _make_loss(self, loss_name: str, weighted: bool):
        # weighted=True => we need elementwise losses
        reduction = 'none' if weighted else 'mean'
        if loss_name == 'mse':
            return nn.MSELoss(reduction=reduction)
        elif loss_name == 'mae':
            return nn.L1Loss(reduction=reduction)
        elif loss_name == 'logloss':  # BCE on probabilities in [0,1]
            return nn.BCELoss(reduction=reduction)
        elif loss_name == 'cross_entropy':  # expects class indices / logits
            return nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_fn: {loss_name}")

    def reset_model(self) -> None:
        """
        Reset the model's weights.
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
        Predict the target values for the given input features.
        INPUTS:
        - X (pd.DataFrame) : Inputs in non-standardized form.
        OUTPUTS:
        - y_pred (np.ndarray): Predicted target values in non-standardized form.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        X_test_df = X.copy()
        # X_test_df = X_test_df.astype(np.float32)

        # -------- standardize features if needed ----------------
        if self.standardize == True:
            X_test_df.loc[:, self.feature_cols] = self.scaler_feature.transform(X[self.feature_cols].values)

        elif isinstance(self.standardize, list):
            stand_features = list(np.array(self.feature_cols)[np.isin(self.feature_cols, self.standardize)])
            stand_targets = list(np.array(self.target_cols)[np.isin(self.target_cols, self.standardize)])
            if len(stand_features) > 0:
                X_test_df.loc[:,stand_features] = self.scaler_feature.transform(X[stand_features].values)

        # -------- build X_test ----------------
        if isinstance(self.feature_cols[0], (list, tuple)):
            seq_len    = len(self.feature_cols)          # time steps
            input_size = len(self.feature_cols[0])       # features/hour
            X_test = np.empty((len(X_test_df), seq_len, input_size), dtype=np.float32)
            for t, cols_t in enumerate(self.feature_cols):
                X_test[:, t, :] = X_test_df[cols_t].values
        else:
            X_test = X_test_df[self.feature_cols].values.astype(np.float32)

        # -------- predict ----------------
        # y_pred = self.model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        y_pred_t = self.model(torch.tensor(X_test, dtype=torch.float32, device=device))
        y_pred = y_pred_t.detach().cpu().numpy()

        # -------- inverse-transform targets if needed ----------------
        if len(stand_targets) > 0:
            y_pred = y_pred.reshape(-1, len(np.array(self.target_cols).flatten()))  # Ensure y_pred is 2D
            y_pred_df = pd.DataFrame(y_pred, columns=np.array(self.target_cols).flatten())
            y_pred_df.loc[:,stand_targets] = self.scaler_target.inverse_transform(y_pred_df[stand_targets])
            if isinstance(self.target_cols[0], (list, tuple)):
                seq_len    = len(self.target_cols)          # time steps
                out_size   = len(self.target_cols[0])       # targets/hour
                Y = np.empty((len(y_pred_df), seq_len, out_size), dtype=np.float32)
                for t, cols_t in enumerate(self.target_cols):
                    Y[:, t, :] = y_pred_df[cols_t].values
            else:
                Y = y_pred_df[self.target_cols].values.astype(np.float32)
            y_pred = Y

        return y_pred

    # Save and load model
    def save_model(self, model_path: str) -> None:
        """
        Save a self-describing checkpoint that can rebuild the model and restore metadata.
        """
        if self.model is None:
            raise ValueError("No model to save. Did you call build_model()?")

        # Build spec must be set by subclasses in build_model()
        build_spec = getattr(self, "_build_spec", None)
        if build_spec is None:
            raise RuntimeError(
                "Subclasses must set self._build_spec in build_model() so the architecture can be reconstructed."
            )

        # Pack scalers (bytes) only if they exist
        def _pickle_or_none(obj):
            try:
                return pickle.dumps(obj) if obj is not None else None
            except Exception:
                return None  # stay resilient if someone attached a weird object

        checkpoint = {
            "format_version": 2,
            "saved_at": datetime.now(UTC).isoformat(),
            "libs": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": StandardScaler.__module__.split('.')[0],  # just to note dependency
            },
            "model": {
                "module": self.__class__.__module__,
                "classname": self.__class__.__name__,
                "build_spec": build_spec,                # {"builder": "build_model", "kwargs": {...}}
                "state_dict": self.model.state_dict(),   # pure tensors – torch.save-safe
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

        # single call; stores only tensors + simple python + bytes (scalers) -> safe across PyTorch versions
        torch.save(checkpoint, model_path)
        if self.verbose:
            print(f"Saved checkpoint to: {model_path}")

    @classmethod
    def load_model(cls, model_path: str, map_location: str = "cpu", verbose: bool = True):
        """
        Classmethod that returns a reconstructed model instance from file.
        Works for any subclass, as long as that subclass set self._build_spec in build_model().
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        # torch.load of pure tensors + primitives is safe across 2.6+
        ckpt = torch.load(model_path, map_location=map_location, weights_only=False)

        mod_name = ckpt["model"]["module"]
        class_name = ckpt["model"]["classname"]
        build_spec = ckpt["model"]["build_spec"]
        state_dict = ckpt["model"]["state_dict"]

        # Dynamically import the original class so we can instantiate the right subclass
        module = importlib.import_module(mod_name)
        klass = getattr(module, class_name)

        # Create an instance (constructor only has verbose, so we're safe)
        obj = klass(verbose=verbose)

        # Restore commonly used attributes (before build)
        data_section = ckpt.get("data", {})
        obj.feature_cols = data_section.get("feature_cols", None)
        obj.target_cols  = data_section.get("target_cols", None)
        obj.standardize  = data_section.get("standardize", False)

        # Rebuild architecture from spec and load weights
        if not build_spec or "builder" not in build_spec or "kwargs" not in build_spec:
            raise RuntimeError("Invalid or missing build_spec in checkpoint – cannot rebuild model.")

        builder_name = build_spec["builder"]
        builder_kwargs = build_spec["kwargs"]

        # Call the builder on the instance (e.g. obj.build_model(**kwargs))
        if not hasattr(obj, builder_name):
            raise AttributeError(f"Target class {class_name} has no builder method '{builder_name}'.")
        getattr(obj, builder_name)(**builder_kwargs)

        # Now load weights
        missing, unexpected = obj.model.load_state_dict(state_dict, strict=False)
        if verbose and (missing or unexpected):
            print(f"[load_model] Missing keys: {missing}")
            print(f"[load_model] Unexpected keys: {unexpected}")

        # Restore scalers if present
        def _unpickle_or_none(b):
            try:
                return pickle.loads(b) if b is not None else None
            except Exception:
                return None

        obj.scaler_feature = _unpickle_or_none(data_section.get("scaler_feature", None))
        obj.scaler_target  = _unpickle_or_none(data_section.get("scaler_target", None))

        # Restore training metadata
        train_section = ckpt.get("train", {})
        obj.optimizer_name = train_section.get("optimizer_name", None)
        obj.loss_fn_name   = train_section.get("loss_fn_name", None)
        obj.val_loss       = train_section.get("val_loss", [])
        obj.num_parameters = train_section.get("num_parameters", None)

        obj.model.eval()
        if verbose:
            print(f"Loaded {class_name} from {model_path}")
            print(f"Rebuilt with {builder_name}(**{list(builder_kwargs.keys())})")

        return obj

    def _gather_val_arrays(self, device=None, loss_name: str = "mse"):
        """
        Collect validation features/targets from self.val_data.
        Returns X_val (N,D), y_val and weights ([N]) with a shape appropriate for the given loss:
          - mse/mae/logloss (BCE): y_val is (N,T) (T can be 1 or >1)
          - cross_entropy: y_val is (N,) integer class indices
                           If stored as one-hot (N,C), we convert to indices via argmax.
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
            Xs.append(xb.detach().cpu())
            Ys.append(yb.detach().cpu())
            Ws.append(wb.detach().cpu())

        X = torch.cat(Xs, dim=0)  # (N, D) or (N, T, D)
        Y = torch.cat(Ys, dim=0)  # (N, T_out) or (N, T, D_out) or (N,) etc.
        W = torch.cat(Ws, dim=0)  # (N,)

        # Flatten inputs if they are sequences
        if X.ndim > 2:
            X = X.view(X.size(0), -1)

        # Targets handling by loss
        loss_name = loss_name.lower()
        if loss_name in ("mse", "mae", "logloss"):
            # Flatten targets over non-batch dims -> (N, T_out)
            if Y.ndim > 2:
                Y = Y.view(Y.size(0), -1)
            elif Y.ndim == 1:
                Y = Y.view(-1, 1)
            # else keep (N, T_out)
            return X.numpy(), Y.numpy(), W.numpy()

        elif loss_name == "cross_entropy":
            # Expect class indices (N,). If we have one-hot (N,C), convert.
            if Y.ndim == 2:
                # If it's one-hot, argmax -> indices
                Y_idx = Y.argmax(dim=1)
            elif Y.ndim == 1:
                Y_idx = Y
            else:
                # Unexpected shape -> flatten last dims to classes and argmax
                Y_idx = Y.view(Y.size(0), -1).argmax(dim=1)
            return X.numpy(), Y_idx.numpy().astype(np.int64), W.numpy()

        else:
            raise ValueError("Unknown loss_name: use 'mse', 'mae', 'logloss', or 'cross_entropy'.")

    @torch.no_grad()
    def _predict_np(self, X_np, batch_size=4096, device=None):
        """
        Forward pass returning raw model outputs as numpy.
        Shape:
          - regression/BCE: (N, T)  (T = number of outputs)
          - cross_entropy:  (N, C)  (C = number of classes)
        """
        self.model.eval()
        if device is None:
            device = next(self.model.parameters()).device
        preds = []
        N = X_np.shape[0]
        for i in range(0, N, batch_size):
            xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32, device=device)
            yb = self.model(xb)
            # ensure 2D (N, T)
            if yb.ndim > 2:
                yb = yb.view(yb.size(0), -1)
            preds.append(yb.detach().cpu())
        Y = torch.cat(preds, dim=0).numpy()
        return Y

    def _loss_np(self, y_true, y_pred, weights=None, loss='mse'):
        """
        Numpy loss to mirror training choices.
        - mse/mae: mean across samples & outputs
        - logloss (BCE): mean BCE over samples & outputs; y_pred expected in [0,1]
        - cross_entropy: y_true class indices (N,), y_pred logits (N,C)
        """
        loss = loss.lower()

        def _reduce_elemwise_loss(loss_elem, w=None):
            """
            loss_elem shape:
            - Regression (MSE/MAE): [B, D] or [B, T, D]
            - BCE: same as regression
            - CE: [B] (already per-sample)
            Returns scalar loss (weighted or unweighted mean over batch).
            """
            if loss_elem.dim() == 1:
                per_sample = loss_elem  # [B]
            else:
                # average across all target dimensions per sample
                per_sample = loss_elem.view(loss_elem.size(0), -1).mean(dim=1)  # [B]

            if w is None:
                return per_sample.mean()
            # weighted mean over batch; avoid div-by-zero
            denom = w.sum().clamp_min(1e-12)
            return (per_sample * w).sum() / denom

        if loss == 'mse':
            return _reduce_elemwise_loss((y_true - y_pred) ** 2, w=weights)

        if loss == 'mae':
            return _reduce_elemwise_loss(np.abs(y_true - y_pred), w=weights)

        if loss == 'logloss':  # BCE
            eps = 1e-10
            # Clip predicted probabilities
            p = np.clip(y_pred, eps, 1.0 - eps)
            # Ensure shapes are compatible
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            bce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
            return _reduce_elemwise_loss(bce, w=weights)

        if loss == 'cross_entropy':
            # y_true: (N,) int class indices
            # y_pred: (N,C) logits or (if your model already applies softmax) probabilities.
            # We'll treat them as logits and apply log_softmax; if they look like probs, it still works numerically.
            # log_softmax for numerical stability
            logits = y_pred
            # subtract max per row
            z = logits - logits.max(axis=1, keepdims=True)
            log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            n = y_true.shape[0]
            # pick the correct class log-prob
            ll = log_probs[np.arange(n), y_true]
            return -_reduce_elemwise_loss(ll, w=weights)

        raise ValueError("loss must be 'mse', 'mae', 'logloss', or 'cross_entropy'")

    def get_feature_importance(
        self,
        method: str = "permutation",     # 'permutation' or 'gradient'
        n_repeats: int = 5,              # for permutation
        loss: str | None = None,         # default: use self.loss_fn_name if set; else 'mse'
        batch_size: int = 4096,
        top_k: int = 20,
        device: str | None = None,
        normalize: bool = True,          # normalize importances (perm: sum=1; grad: max=1)
        return_df: bool = False
    ):
        """
        Compute & plot feature importance for the MLP with loss matching training:
          - 'mse', 'mae', 'logloss' (BCE), 'cross_entropy'
        """
        if device is None:
            device = next(self.model.parameters()).device
        if loss is None:
            loss = getattr(self, "loss_fn_name", "mse")

        # Resolve flat feature names
        if isinstance(self.feature_cols[0], (list, tuple)):
            feat_names = [f"{t}:{name}" for t, cols_t in enumerate(self.feature_cols) for name in cols_t]
        else:
            feat_names = list(map(str, self.feature_cols))

        method = method.lower()
        loss_l = loss.lower()

        # --- PERMUTATION IMPORTANCE ---
        if method == "permutation":
            X_val, y_val = self._gather_val_arrays(device=device, loss_name=loss_l)
            y_pred = self._predict_np(X_val, batch_size=batch_size, device=device)
            base = self._loss_np(y_val, y_pred, loss=loss_l)

            rng = np.random.default_rng(42)
            importances = np.zeros(X_val.shape[1], dtype=float)

            X_work = X_val.copy()
            for j in range(X_val.shape[1]):
                deltas = []
                for _ in range(n_repeats):
                    rng.shuffle(X_work[:, j])
                    y_perm = self._predict_np(X_work, batch_size=batch_size, device=device)
                    L = self._loss_np(y_val, y_perm, loss=loss_l)
                    deltas.append(L - base)
                    X_work[:, j] = X_val[:, j]   # restore column
                importances[j] = np.mean(deltas)

            if normalize:
                s = importances.sum()
                if s > 0:
                    importances = importances / s

            importance = {f:i for f, i in zip(feat_names, importances)}

            return importance

        # --- GRADIENT SALIENCY (independent of loss) ---
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

                yhat = self.model(xb)  # (B, T) or (B, C)
                scalar = yhat.mean()
                self.model.zero_grad(set_to_none=True)
                scalar.backward()

                g = xb.grad.detach().abs().mean(dim=0)  # (D,)
                grads_sum = g if grads_sum is None else grads_sum + g
                count += 1

            grads_mean = (grads_sum / max(count, 1)).cpu().numpy()
            if normalize and grads_mean.max() > 0:
                grads_mean = grads_mean / grads_mean.max()

            importance = {f:i for f, i in zip(feat_names, grads_mean)}

            return importance

        else:
            raise ValueError("method must be 'permutation' or 'gradient'")

    # ---------- main: loss-aware feature importance ----------
    def plot_feature_importance(
        self,
        method: str = "permutation",     # 'permutation' or 'gradient'
        n_repeats: int = 5,              # for permutation
        loss: str | None = None,         # default: use self.loss_fn_name if set; else 'mse'
        batch_size: int = 4096,
        top_k: int = 20,
        device: str | None = None,
        normalize: bool = True,          # normalize importances (perm: sum=1; grad: max=1)
        return_df: bool = False
    ):
        """
        Compute & plot feature importance for the MLP with loss matching training:
          - 'mse', 'mae', 'logloss' (BCE), 'cross_entropy'
        """
        importances_dict = self.get_feature_importance(
            method=method,
            n_repeats=n_repeats,
            loss=loss,
            batch_size=batch_size,
            top_k=top_k,
            device=device,
            normalize=normalize,
            return_df=return_df)

        feat_names = list(importances_dict.keys())
        importances = list(importances_dict.values())

        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)

        if loss is None:
            loss = getattr(self, "loss_fn_name", "mse")
        loss_l = loss.lower()
        

        k = min(top_k, len(imp_df))
        plt.figure(figsize=(8, max(3.5, 0.4 * k)))
        plt.barh(imp_df["Feature"][:k][::-1], imp_df["Importance"][:k][::-1])
        plt.xlabel(f"Permutation Importance ({loss_l})" + (" (normalized)" if normalize else ""))
        plt.title(f"MLP Feature Importance (Permutation, repeats={n_repeats})")
        plt.tight_layout()
        plt.show()

        return imp_df if return_df else None




class xgboostModel(GeneratorFailureProbabilityInference):
    """
    Class for inferring the probability of generator failure with an XGBoost model.
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)

    def build_model(self, max_depth : int, eta : float, gamma : float, reg_lambda : float,
                    num_boost_round: int = 100,
                    feature_cols: list[str] = None, target_cols: list[str] = None,
                    eval_metric: str = 'rmse', objective: str = 'reg:logistic',
                    early_stopping_rounds: int = 10,
                    subsample : float = 1,
                    device = 'cpu') -> None:
        """
        Build the XGBoost model architecture.
        INPUTS:
        - max_depth (int): Maximum depth of a tree.
        - eta (float): Step size shrinkage used in update to prevents overfitting.
        - gamma (float): Minimum loss reduction required to make a further partition on a leaf node.
        - reg_lambda (float): L2 regularization term on weights.
        - num_boost_round (int): Number of boosting rounds.
        - feature_cols (list[str]): List of feature column names.
        - target_cols (list[str]): List of target column names.
        - subsample (float): Subsample ratio of the training instances.
        - device (str): Device to run the model on (e.g., 'cpu' or 'gpu').
        """
        self.max_depth = max_depth
        self.eta = eta
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.num_boost_round = num_boost_round
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.eval_metric = eval_metric
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample

        self.model = xgb.XGBRegressor(max_depth=max_depth, 
                                      eta=eta, gamma=gamma, 
                                      reg_lambda=reg_lambda,
                                      n_estimators=num_boost_round,
                                      subsample=subsample,
                                      eval_metric=eval_metric,
                                      objective=objective,
                                      early_stopping_rounds=early_stopping_rounds,
                                      verbosity=1 if self.verbose else 0,
                                      device=device)
        
        # record how to rebuild this architecture
        self._build_spec = {
            "builder": "build_model",
            "kwargs": {
                "max_depth": max_depth,
                "eta": eta,
                "gamma": gamma,
                "reg_lambda": reg_lambda,
                "num_boost_round": num_boost_round,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "eval_metric": eval_metric,
                "objective": objective,
                "early_stopping_rounds": early_stopping_rounds,
                "subsample": subsample,
            }
        }
        
        if self.verbose:
            print(f"Model architecture: {self.model}")

    def train_model(self, weights_data = False) -> None:
        """
        Train the model on the training dataset.
        INPUTS:
        - weights_data (bool): if True, use sample weights for training. Default is False.
        """

        if weights_data:
            weights = self.train_data['Data_weight'].values
        else:
            weights = np.ones(len(self.train_data))



        X_train = self.train_data[self.feature_cols]#.values
        y_train = self.train_data[self.target_cols]#.values
        X_val = self.val_data[self.feature_cols]#.values
        y_val = self.val_data[self.target_cols]#.values

        # dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
        # dtest = xgb.DMatrix(X_val, label=y_val)

        self.model.fit(X_train, y_train, 
                       eval_set=[(X_val, y_val)], sample_weight=weights)
    
    def reset_model(self) -> None:
        """
        Reset the model's weights.
        """
        if self.model is not None:
            self.model = xgb.XGBRegressor(max_depth=self.max_depth, 
                                      eta=self.eta, gamma=self.gamma, 
                                      lambda_=self.reg_lambda,
                                      num_boost_round=self.num_boost_round,
                                      subsample=self.subsample,
                                      verbose=self.verbose)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the target values for the given input features.
        INPUTS:
        - X (pd.DataFrame) : Inputs in non-standardized form.
        OUTPUTS:
        - y_pred (np.ndarray): Predicted target values in non-standardized form.
        """
        if self.model is None:
            raise ValueError("Model not built. Please build the model first.")

        X_test_df = X.copy()

        # -------- standardize features if needed ----------------
        if self.standardize == True:
            X_test_df.loc[:, self.feature_cols] = self.scaler_feature.transform(X[self.feature_cols].values)

        elif isinstance(self.standardize, list):
            stand_features = list(np.array(self.feature_cols)[np.isin(self.feature_cols, self.standardize)])
            stand_targets = list(np.array(self.target_cols)[np.isin(self.target_cols, self.standardize)])
            if len(stand_features) > 0:
                X_test_df.loc[:,stand_features] = self.scaler_feature.transform(X[stand_features].values)

        # -------- build X_test ----------------
        if isinstance(self.feature_cols[0], (list, tuple)):
            seq_len    = len(self.feature_cols)          # time steps
            input_size = len(self.feature_cols[0])       # features/hour
            X_test = np.empty((len(X_test_df), seq_len, input_size), dtype=np.float32)
            for t, cols_t in enumerate(self.feature_cols):
                X_test[:, t, :] = X_test_df[cols_t].values
        else:
            X_test = X_test_df[self.feature_cols].values.astype(np.float32)

        # X_test = xgb.DMatrix(X_test)
        # -------- predict ----------------
        y_pred = self.model.predict(X_test)

        # -------- inverse-transform targets if needed ----------------
        if len(stand_targets) > 0:
            y_pred = y_pred.reshape(-1, len(np.array(self.target_cols).flatten()))  # Ensure y_pred is 2D
            y_pred_df = pd.DataFrame(y_pred, columns=np.array(self.target_cols).flatten())
            y_pred_df.loc[:,stand_targets] = self.scaler_target.inverse_transform(y_pred_df[stand_targets])
            if isinstance(self.target_cols[0], (list, tuple)):
                seq_len    = len(self.target_cols)          # time steps
                out_size   = len(self.target_cols[0])       # targets/hour
                Y = np.empty((len(y_pred_df), seq_len, out_size), dtype=np.float32)
                for t, cols_t in enumerate(self.target_cols):
                    Y[:, t, :] = y_pred_df[cols_t].values
            else:
                Y = y_pred_df[self.target_cols].values.astype(np.float32)
            y_pred = Y

        return y_pred
        













        
        X_test = X[self.feature_cols].values.astype(np.float32)
        y_pred = self.model.predict(X_test)
        
        # Inverse transform if necessary
        if hasattr(self, 'scaler_target'):
            y_pred = self.scaler_target.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        return y_pred

    def save_model(self, model_path: str) -> None:
        """
        Save a self-describing checkpoint for XGBoost: model spec, booster bytes,
        scalers, and metadata. Single-file, like the MLP saver.
        """
        if self.model is None:
            raise ValueError("No model to save. Did you call build_model()?")

        build_spec = getattr(self, "_build_spec", None)
        if build_spec is None:
            raise RuntimeError("xgboostModel must set self._build_spec in build_model().")

        # Safely pickle scalers if present
        def _pickle_or_none(obj):
            try:
                return pickle.dumps(obj) if obj is not None else None
            except Exception:
                return None

        # Grab booster bytes if the model is fitted
        booster_raw = None
        try:
            booster = self.model.get_booster()
            booster_raw = booster.save_raw()  # bytes
        except Exception:
            # not fitted yet; that’s okay, we still save the spec
            booster_raw = None

        # Optional training/eval metadata
        evals_result = None
        try:
            evals_result = self.model.evals_result()
        except Exception:
            pass

        checkpoint = {
            "format_version": 1,
            "saved_at": datetime.now(UTC).isoformat(),
            "libs": {
                "xgboost": xgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": StandardScaler.__module__.split('.')[0],
            },
            "model": {
                "module": self.__class__.__module__,
                "classname": self.__class__.__name__,
                "build_spec": build_spec,                     # {"builder": "...", "kwargs": {...}}
                "xgb": {
                    "sk_params": self.model.get_params(deep=False),  # sklearn estimator params
                    "booster_raw": booster_raw,               # bytes or None
                    "best_iteration": getattr(self.model, "best_iteration", None),
                    "best_score": getattr(self.model, "best_score", None),
                    "evals_result": evals_result,             # dict or None
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

        # One file, same as your MLP
        torch.save(checkpoint, model_path)
        if self.verbose:
            print(f"Saved XGBoost checkpoint to: {model_path}")

    @classmethod
    def load_model(cls, model_path: str, verbose: bool = True):
        """
        Reconstruct an xgboostModel instance from a single-file checkpoint.
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

        # Dynamically import and instantiate correct subclass
        module = importlib.import_module(mod_name)
        klass = getattr(module, class_name)
        obj = klass(verbose=verbose)

        # Restore common attributes (before build)
        obj.feature_cols = data_section.get("feature_cols", None)
        obj.target_cols  = data_section.get("target_cols", None)
        obj.standardize  = data_section.get("standardize", False)

        # Rebuild estimator from build spec
        if not build_spec or "builder" not in build_spec or "kwargs" not in build_spec:
            raise RuntimeError("Invalid or missing build_spec in checkpoint – cannot rebuild model.")
        builder_name = build_spec["builder"]
        builder_kwargs = build_spec["kwargs"]
        if not hasattr(obj, builder_name):
            raise AttributeError(f"Target class {class_name} has no builder method '{builder_name}'.")
        getattr(obj, builder_name)(**builder_kwargs)

        # Ensure sklearn params are identical (defensive)
        sk_params = xgb_section.get("sk_params", {})
        if sk_params:
            obj.model.set_params(**sk_params)

        # --- restore booster if available ---
        booster_raw = xgb_section.get("booster_raw", None)
        if booster_raw is not None:
            booster = xgb.Booster()
            booster.load_model(bytearray(booster_raw))  # load from memory
            obj.model._Booster = booster

            # optionally set n_features_in_ so sklearn doesn't complain
            try:
                if obj.feature_cols is not None:
                    if isinstance(obj.feature_cols[0], (list, tuple)):
                        n_features = len(obj.feature_cols) * len(obj.feature_cols[0])
                    else:
                        n_features = len(obj.feature_cols)
                    obj.model.n_features_in_ = n_features
            except Exception:
                pass

        # --- DO NOT assign to read-only properties on the estimator ---
        # Keep these as metadata on the wrapper object instead.
        obj.best_iteration = xgb_section.get("best_iteration", None)
        obj.best_score     = xgb_section.get("best_score", None)
        obj._evals_result  = xgb_section.get("evals_result", None)

        # Restore scalers
        def _unpickle_or_none(b):
            try:
                return pickle.loads(b) if b is not None else None
            except Exception:
                return None
        obj.scaler_feature = _unpickle_or_none(data_section.get("scaler_feature", None))
        obj.scaler_target  = _unpickle_or_none(data_section.get("scaler_target", None))

        # Training metadata
        obj.early_stopping_rounds = train_section.get("early_stopping_rounds", None)

        if verbose:
            print(f"Loaded {class_name} from {model_path}")
            print(f"Rebuilt with {builder_name}(**{list(builder_kwargs.keys())})")
            if booster_raw is None:
                print("Note: booster was not saved (model likely not fitted yet).")

        return obj

    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance from the XGBoost model.
        
        Parameters:
        - importance_type: Type of importance to retrieve ('weight', 'gain', 'cover').
        
        Returns:
        - importance_df: DataFrame with feature names and their importance scores.
        """
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)
        return importance

    def plotFeatureImportance(self, importance_criterions=['weight', 'gain', 'cover'], n_features=10):
        """
        Plots the feature importance of the model.
        
        Parameters:
        - model: Trained XGBoost model.
        - features_names: List of feature names.
        - n_features: Number of top features to display.
        
        Returns:
        - None
        """
        fig, axs = plt.subplots(1, len(importance_criterions), figsize=(6*len(importance_criterions), n_features))


        for i, criterion in enumerate(importance_criterions):

            importance = self.get_feature_importance(importance_type=criterion)
            importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False).reset_index(drop=True)

            axs[i].barh(importance_df['Feature'][:n_features], importance_df['Importance'][:n_features], color='skyblue')
            axs[i].set_xlabel('Importance')
            axs[i].set_title(f'Feature Importance - {criterion}')
            axs[i].invert_yaxis()

        plt.tight_layout()
        plt.show()




# Grid search utilities


def _expand_grid(param_grid: dict):
    """
    Turn a dict of lists into a list of dicts (Cartesian product).
    Example: {"a":[1,2], "b":[10]} -> [{"a":1,"b":10}, {"a":2,"b":10}]
    """
    keys = list(param_grid.keys())
    vals = [param_grid[k] if isinstance(param_grid[k], (list, tuple)) else [param_grid[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def _val_loss_numpy(model, metric: str = "mse"):
    """
    Compute validation loss for any model that has:
      - model.val_data  (DataFrame or torch Dataset)
      - model.feature_cols, model.target_cols
      - model.predict(X_df) -> np.ndarray
    Falls back to numpy metrics; matches training losses:
      'mse', 'mae', 'logloss' (BCE over probabilities in [0,1]).
    """
    metric = metric.lower()

    # Build a DataFrame view for val set (works for MLP and XGB paths)
    if hasattr(model, "val_data"):
        if isinstance(model.val_data, pd.DataFrame):
            val_df = model.val_data
        else:
            # Torch Subset/Dataset path (MLP). Reconstruct a DataFrame.
            # We'll reuse the model’s helpers by emulating predict with DataLoader,
            # but simpler: pull from original split in prepare_data when MLP path used.
            # For MLP we already rely on model.val_loss, so return min of it.
            if hasattr(model, "val_loss") and len(model.val_loss) > 0:
                return float(np.min(model.val_loss))
            raise ValueError("val_data is not a DataFrame and val_loss is empty; cannot score.")
    else:
        raise ValueError("Model has no val_data. Did you call prepare_data()?")

    X_cols = list(np.unique(np.array(model.feature_cols).flatten()))
    y_cols = list(np.unique(np.array(model.target_cols).flatten()))

    X_val = val_df[X_cols].copy()
    y_true = val_df[y_cols].to_numpy(dtype=np.float64)
    y_pred = model.predict(X_val).reshape(y_true.shape).astype(np.float64)

    if metric == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "logloss":  # BCE
        eps = 1e-7
        p = np.clip(y_pred, eps, 1.0 - eps)
        # Ensure shapes align
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        bce = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return float(np.mean(bce))

    raise ValueError("metric must be 'mse', 'mae', or 'logloss'")

def _already_done_df(csv_path: str):
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "level","model_name","build_params","train_params",
        "median_min_val_loss","timestamp"
    ])

def _row_key(level_name, model_name, build_params, train_params):
    # Create a deterministic key for resume-ability
    return (
        str(level_name),
        str(model_name),
        json.dumps(build_params, sort_keys=True).replace(",", ";"),
        json.dumps(train_params,  sort_keys=True).replace(",", ";"),
    )

def successive_halving_search(
    model_specs: list,
    # Each item in model_specs:
    # {
    #   "name": "MLP" or "XGB-something",
    #   "constructor": callable -> returns a fresh model instance (e.g., im.MLP, im.xgboostModel),
    #   "build_grid": dict of lists for build_model kwargs (architecture/hyperparams),
    #   "train_grid": dict of lists for train_model kwargs (optimizer, loss_fn, epochs, ...),
    #   "common_build": dict (fixed build_model kwargs, e.g., feature_cols, target_cols),
    #   "common_train": dict (fixed train_model kwargs, e.g., weights_data=True, device="cpu")
    # }
    data: pd.DataFrame,
    standardize: list | bool,
    result_csv: str,
    train_ratio: float = 0.8,
    val_ratio: float   = 0.2,
    val_metric_per_model: dict | None = None,  # e.g., {"xgb": "logloss", "mlp": "logloss"}
    levels: list[dict] | None = None,
    top_keep_ratio: float = 0.33,
    resume: bool = True
    ):
    """
    Generic successive-halving for your MLP & XGBoost wrappers.

    - model_specs: list of model search specifications (see structure above)
    - levels: [{"name":..., "epochs":..., "data_cap": int|None}, ...]
    - Writes/reads a CSV log at result_csv.
    """
    if levels is None:
        N = len(data)
        levels = [
            {"name":"L1-fast",   "epochs": 150, "data_cap": int(N*0.4)},
            {"name":"L2-medium", "epochs": 500, "data_cap": int(N*0.8)},
            {"name":"L3-full",   "epochs": 2000,"data_cap": None},
        ]

    # Build all candidate tuples (model_index, build_params, train_params)
    all_candidates = []
    for i, spec in enumerate(model_specs):
        for build_params in _expand_grid(spec.get("build_grid", {})):
            for train_params in _expand_grid(spec.get("train_grid", {})):
                all_candidates.append((i, build_params, train_params))
    print(f"Number of candidates : {len(all_candidates)}")
    # Resume logic
    done_df = _already_done_df(result_csv)
    
    done_keys = set()
    if resume and len(done_df):
        for _, r in done_df.iterrows():
            done_keys.add((r["level"], r["model_name"], r["build_params"], r["train_params"]))
    

    survivors = all_candidates[:]

    for li, level in enumerate(levels):
        scored = []
        for (mi, build_params, train_params) in survivors:
            spec = model_specs[mi]
            mname = spec["name"]
            level_epochs = level["epochs"]
            data_cap = level["data_cap"]
            sub_data = data.iloc[:data_cap].copy() if data_cap is not None else data.copy()

            # Merge fixed params
            build_kw = dict(spec.get("common_build", {}))
            build_kw.update(build_params)

            train_kw = dict(spec.get("common_train", {}))
            train_kw.update(train_params)
            # cap epochs at this level
            if "epochs" in train_kw:
                train_kw["epochs"] = min(int(train_kw["epochs"]), int(level_epochs))

            # Resume skip?
            key = _row_key(level["name"], mname, build_kw, train_kw)

            if resume and key in done_keys:
                print(f"Skipping already-done: level={level['name']} model={mname} build={build_kw} train={train_kw}")
                # Pull previous score
                prev = done_df.loc[
                    (done_df["level"] == level["name"]) &
                    (done_df["model_name"] == mname) &
                    (done_df["build_params"] == key[2]) &
                    (done_df["train_params"] == key[3])
                ]
                score = float(prev["median_min_val_loss"].iloc[0])
                scored.append((mi, build_params, train_params, score))
                continue


            # ----- Build, prepare, train, score -----
            model = spec["constructor"]()
            # 1) build
            model.build_model(**build_kw)
            # 2) prepare data
            model.prepare_data(
                data=sub_data,
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=0.0,
                standardize=standardize
            )
            # 3) reset + train
            if hasattr(model, "reset_model"):
                model.reset_model()
            model.val_loss = []

            # train with only the kwargs accepted by each class API
            # (this works because both classes ignore extra kwargs they don't take)
            model.train_model(**train_kw)

            # 4) score
            # - For MLP: use min(val_loss) directly (fast)
            # - For XGB: compute validation loss with chosen metric
            metric = None
            if val_metric_per_model:
                # allow key by exact name, lowercase name, or class name
                metric = (
                    val_metric_per_model.get(mname) or
                    val_metric_per_model.get(mname.lower()) or
                    val_metric_per_model.get(model.__class__.__name__) or
                    val_metric_per_model.get(model.__class__.__name__.lower())
                )

            if hasattr(model, "val_loss") and len(model.val_loss) > 0:
                score = float(np.min(model.val_loss))
            else:
                score = _val_loss_numpy(model, metric or "mse")

            scored.append((mi, build_params, train_params, score))

            # 5) append row to CSV
            row = {
                "level": level["name"],
                "model_name": mname,
                "build_params": json.dumps(build_kw, sort_keys=True).replace(",", ";"),
                "train_params": json.dumps(train_kw, sort_keys=True).replace(",", ";"),
                "median_min_val_loss": score,
                "timestamp": _dt.now().isoformat()
            }
            # create dir once
            if result_csv is not None:
                os.makedirs(os.path.dirname(result_csv), exist_ok=True)
                # write header if file does not exist
                write_header = not os.path.exists(result_csv)
                with open(result_csv, "a") as f:
                    if write_header:
                        f.write(",".join(row.keys()) + "\n")
                    f.write(",".join(map(str, row.values())) + "\n")

        # select survivors
        scored.sort(key=lambda t: t[3])
        keep = max(1, int(math.ceil(len(scored) * top_keep_ratio)))
        survivors = [(mi, bp, tp) for (mi, bp, tp, _) in scored[:keep]]

        # final output
        if li == len(levels) - 1:
            return scored[:keep]


# Concerns
# - When calling a new model, is it really new or taking the previous one??