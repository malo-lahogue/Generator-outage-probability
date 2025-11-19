from __future__ import annotations

# ── Environment settings (must come before heavy libs) ───────────────────────────
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

# ── Standard Library ────────────────────────────────────────────────────────────
import csv
import datetime
import importlib
import itertools
import json
import math
import pickle
import warnings
from collections import deque
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Literal, List,
    Optional, Sequence, Tuple, Union
)

# ── Third-Party: Core scientific stack ──────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── PyTorch (must precede sklearn sometimes for MKL thread settings) ────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
torch.set_num_threads(1)

# ── Scikit-Learn ────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

# ── XGBoost ─────────────────────────────────────────────────────────────────────
import xgboost as xgb




# Set seed for reproducibility
# np.random.seed(0)
torch.manual_seed(0)

def true_prob_focal_loss(logits:torch.tensor, gamma:float):
    """
    Correct the biased probability estimates from focal loss.
    """
    with torch.no_grad():
        # assert torch.isfinite(logits).all()
        p = F.softmax(logits, dim=1)
        p = torch.clamp(p, min=1e-12, max=1.0)  # numerical stability
        one_minus_p = torch.clamp(1 - p, min=1e-12, max=1.0)

        denom = one_minus_p ** gamma - gamma * one_minus_p ** (gamma - 1) * p * torch.log(p)
        denom = torch.clamp(denom, min=1e-12)
        pt = p / denom
        pt = pt / pt.sum(dim=1, keepdim=True)
        pt = torch.clamp(pt, min=1e-12, max=1.0)  # numerical stability
    return pt



def focal_loss(
    logits: torch.Tensor | None = None,
    probs: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
):
    """
    Multi-class softmax focal loss. Input either logits or softmax probabilities.
    - logits: (N, C)
    - probs: (N, C) softmax probabilities
    - targets: (N,) integer labels in {0,1,2}
    - alpha: weight for class imbalance
    - gamma: focusing parameter
    """
    # Compute softmax probabilities
    if targets is None:
        raise ValueError("Targets must be provided for focal loss computation.")
    if logits is None and probs is None:
        raise ValueError("Either logits or probs must be provided.")
    if logits is not None and probs is not None:
        raise ValueError("Provide only one of logits or probs, not both.")

    if probs is None:
        # assert torch.isfinite(logits).all()
        probs = F.softmax(logits, dim=1)            # (N, C)

    pt = probs[torch.arange(len(targets)), targets]  # (N,)
    pt = torch.clamp(pt, min=1e-12, max=1.0)  # numerical stability

    a = alpha[torch.arange(len(targets)), targets]  # (N,)

    # Focal loss formula
    loss = -a * (1 - pt) ** gamma * torch.log(pt + 1e-12)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

class GeneratorFailureProbabilityInference:
    """
    Base class providing common data preparation, standardization, plotting helpers,
    and save/load utilities for surrogate models (e.g., MLP, XGBoost).

    Typical workflow
    ----------------
    1) Subclass builds a model and sets `self.feature_cols` and `self.target_cols`.
    2) Call `prepare_data(...)` to split and (optionally) standardize.
    3) Call subclass `train_model(...)`.
    4) Call subclass `predict(X)`.

    Attributes set by `prepare_data`
    --------------------------------
    - dataset_df : pd.DataFrame
        Full dataset copy after any reweighting / standardization.
    - train_data, val_data
        Either pandas DataFrames or torch Datasets (for MLP).
    - _train_idx, _val_idx : list[int]
        Indices used for the train/val split.
    - scaler_feature : StandardScaler | None
        Feature scaler (if standardization was requested).
    - scaler_target : StandardScaler | None
        Target scaler (only for continuous/regression targets).
    - standardize : bool | list[str]
        Standardization config passed in.
    - train_ratio, val_ratio : float
        Ratios used for splitting.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

        # model & training state
        self.model: Optional[object] = None
        self.val_loss: list[float] = []

        # data-related attributes
        self.dataset_df: Optional[pd.DataFrame] = None
        self.train_data = None
        self.val_data = None
        self._train_idx: Optional[list[int]] = None
        self._val_idx: Optional[list[int]] = None

        # column specs (must be set by subclasses before prepare_data)
        self.feature_cols: Optional[Sequence[str]] = None
        self.target_cols: Optional[Sequence[str]] = None

        # standardization state
        self.scaler_feature: Optional[StandardScaler] = None
        self.scaler_target: Optional[StandardScaler] = None
        self.standardize: Union[bool, list[str]] = False

        # split ratios
        self.train_ratio: float = 0.0
        self.val_ratio: float = 0.0

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------

    def prepare_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        standardize: Union[bool, list[str]] = False,
        reweight_train_data_density: bool | str = False,
        seed: int = 42,
    ) -> None:
        """
        Split data into train/val (no test here), optionally standardize, and reweight samples.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset with features, targets, and optionally 'Data_weight'.
        train_ratio : float
            Proportion of data allocated to training.
        val_ratio : float
            Proportion allocated to validation.
            Note: train_ratio + val_ratio must be <= 1.0.
        standardize : bool | list[str]
            If True:
                - Standardize all feature columns in `self.feature_cols`.
                - Standardize continuous (non-integer) target columns in `self.target_cols`.
            If list[str]:
                - Standardize only those columns (among features/targets) whose names
                  are in the list and exist in `data`.
            If False:
                - No standardization is applied.
        reweight_train_data_density : bool | str
            If False:
                - No density-based reweighting.
            If str:
                - Name of a numeric column used for 1/density reweighting on the
                  *training* portion via KernelDensity.
        seed : int
            Random seed for data shuffling/splitting.

        Side effects
        ------------
        Sets:
            - self.dataset_df
            - self.train_data, self.val_data
            - self._train_idx, self._val_idx
            - self.scaler_feature, self.scaler_target
            - self.standardize, self.train_ratio, self.val_ratio
        """
        # --- guards on required attributes ---
        for attr in ("feature_cols", "target_cols"):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise AttributeError(
                    f"{self.__class__.__name__}.prepare_data requires '{attr}' to be set "
                    "(usually done in build_model)."
                )

        if any(r < 0 for r in (train_ratio, val_ratio)):
            raise ValueError("train_ratio and val_ratio must be non-negative.")
        if train_ratio + val_ratio > 1.0 + 1e-9:
            raise ValueError("train_ratio + val_ratio must be ≤ 1.0.")

        # --- clone dataset and store config ---
        ds = data.copy()
        self.dataset_df = ds
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.standardize = standardize

        N = len(ds)

        # --- deterministic split (random permutation, fixed seed) ---
        if train_ratio + val_ratio == 0:
            # degenerate case: everything to "train", none to "val"
            train_size = N
            val_size = 0
        else:
            frac = train_ratio / max(train_ratio + val_ratio, 1e-12)
            train_size = int(round(N * frac))
            val_size = N - train_size

        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=rng).tolist()
        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size + val_size]

        self._train_idx = train_idx
        self._val_idx = val_idx

        # --- optional density-based reweighting on training data ---
        if reweight_train_data_density:
            if not isinstance(reweight_train_data_density, str):
                raise ValueError(
                    "'reweight_train_data_density' must be False or a column name (str)."
                )
            col = reweight_train_data_density
            if col not in ds.columns:
                raise KeyError(
                    f"Column '{col}' not found in data for density reweighting."
                )

            print(f"Reweighting training data by 1/density of column '{col}'...")

            X = ds.iloc[train_idx][[col]].to_numpy()

            # simple heuristic bandwidth (avoid 0)
            bw = max((X.max() - X.min()) / 10.0, 1e-6)
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(X)

            # grid for interpolation
            X_grid = np.quantile(X.flatten(), np.linspace(0, 1, 100)).reshape(-1, 1)
            log_dens = kde.score_samples(X_grid)
            density = np.exp(log_dens)

            raw_vals = X.flatten()
            interp_dens = np.interp(raw_vals, X_grid.flatten(), density)
            interp_weights = 1.0 / (interp_dens + 1e-10)
            interp_weights /= np.mean(interp_weights)

            if "Data_weight" not in ds.columns:
                ds["Data_weight"] = 1.0
            ds.loc[train_idx, "Data_weight"] = (
                ds.loc[train_idx, "Data_weight"].to_numpy() * interp_weights
            ).astype(np.float32)

        # --- standardization setup ---
        self.scaler_feature = None
        self.scaler_target = None

        if standardize is True or isinstance(standardize, list):
            self.scaler_feature = StandardScaler()
            self.scaler_target = StandardScaler()

            present_cols = set(ds.columns)

            # which features to standardize
            if standardize is True:
                stand_features = [
                    f for f in self.feature_cols if f in present_cols  # type: ignore[arg-type]
                ]
            else:
                stand_features = [
                    f for f in self.feature_cols  # type: ignore[arg-type]
                    if f in standardize and f in present_cols
                ]

            # which targets to standardize (only continuous / non-integer)
            if standardize is True:
                candidate_targets = [
                    t for t in self.target_cols  # type: ignore[arg-type]
                    if t in present_cols
                ]
            else:
                candidate_targets = [
                    t for t in self.target_cols  # type: ignore[arg-type]
                    if t in standardize and t in present_cols
                ]

            stand_targets: list[str] = []
            for t in candidate_targets:
                # treat integer-typed columns as class labels -> do NOT standardize
                if np.issubdtype(ds[t].dtype, np.integer):
                    continue
                stand_targets.append(t)

            # fit on training slice only (to avoid leakage)
            if stand_features:
                self.scaler_feature.fit(ds.iloc[train_idx][stand_features])
                ds.loc[:, stand_features] = self.scaler_feature.transform(ds[stand_features])
            if stand_targets:
                self.scaler_target.fit(ds.iloc[train_idx][stand_targets])
                ds.loc[:, stand_targets] = self.scaler_target.transform(ds[stand_targets])
            else:
                # if we never standardize targets, keep scaler_target = None to avoid misuse
                self.scaler_target = None

            # persist standardized dataset
            self.dataset_df = ds

        # --- construct train / val splits ---
        if self.__class__.__name__ == "MLP":
            # use torch Dataset wrapper for MLP
            ds_torch = GeneratorDataset(
                ds,
                feature_cols=list(self.feature_cols),   # type: ignore[arg-type]
                target_cols=list(self.target_cols),     # type: ignore[arg-type]
            )
            self.train_data = torch.utils.data.Subset(ds_torch, train_idx)
            self.val_data = torch.utils.data.Subset(ds_torch, val_idx)
        else:
            # default: pandas DataFrames
            self.train_data = ds.iloc[train_idx].reset_index(drop=True)
            self.val_data = ds.iloc[val_idx].reset_index(drop=True)

   # -------------------------------------------------------------------------
    # Plotting helpers
    # -------------------------------------------------------------------------

    def plot_validation_loss(self, y_scale: str = "linear") -> None:
        """
        Plot the validation loss over epochs.

        Parameters
        ----------
        y_scale : str
            Matplotlib y-scale, e.g. 'linear' or 'log'.

        Returns
        -------
        None
        """
        if not self.val_loss:
            raise ValueError("No validation loss recorded yet (val_loss is empty).")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.val_loss)
        ax.set_yscale(y_scale)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Validation Loss Over Epochs")
        plt.tight_layout()
        plt.show()

    def plot_train_test_split(self) -> None:
        """
        Visualize the distribution of target values in train vs test (simple strip plot).

        Requires:
            - self.train_data to be prepared.
            - Optional self.test_data for the test series.

        Returns
        -------
        None
        """
        if not hasattr(self, "train_data") or self.train_data is None:
            raise ValueError("No train_data. Call prepare_data first.")

        y_train = self._y_from_any(self.train_data)

        if getattr(self, "test_data", None) is not None and len(self.test_data) > 0:
            y_test = (
                self.test_data[self.target_cols]  # type: ignore[index]
                .to_numpy()
                .reshape(len(self.test_data), -1)
                .squeeze()
            )
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
        plt.tight_layout()
        plt.show()

    # (Not revised yet)

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

    def plot_feature_mapping(self, max_features: int = 10, test_data: pd.DataFrame | None = None, n_width=5) -> None:
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
        rows = int(np.ceil(max_features / n_width))
        fig, axs = plt.subplots(rows, n_width, figsize=(int(2.4*n_width), 3 * rows))
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


    # ---------------------- internal helpers ----------------------
    def _ensure_test_df(self, test_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Internal: pick a test DataFrame and drop NaNs."""
        if test_data is None:
            if getattr(self, "test_data", None) is None:
                raise ValueError(
                    "Test data not prepared. Provide test_data "
                    "or set self.test_data before calling this method."
                )
            df = self.test_data.copy()
        else:
            df = test_data.copy()
        return df.dropna()

    def _y_from_any(self, split) -> np.ndarray:
        """
        Internal: get a 1D y array from either:
        - a pandas DataFrame (XGB-like models), or
        - a torch Dataset / Subset (MLP).
        """
        # torch Dataset / Subset path
        if isinstance(split, torch.utils.data.Dataset):
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

        # pandas DataFrame path
        if not isinstance(split, pd.DataFrame):
            raise TypeError("split must be a torch Dataset or a pandas DataFrame.")
        y = split[self.target_cols].to_numpy()  # type: ignore[index]
        if y.ndim > 1:
            y = y.reshape(y.shape[0], -1)
        return y.squeeze()

    def _predict_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Internal: call self.predict on a DataFrame and return flattened predictions.
        """
        if self.feature_cols is None:
            raise AttributeError("feature_cols not set; build_model must set it first.")

        X_cols = list(dict.fromkeys(self.feature_cols))  # keep order, drop dups
        X = df[X_cols]
        y_pred = self.predict(X)
        if y_pred.ndim > 1:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
        return y_pred.squeeze()
    
class GeneratorDataset(Dataset):
    """
    Dataset returning consistent (X, y, w) triples.

    Modes
    -----
    classification:
        - target_cols = exactly one column (integer codes)
        - y is a scalar int64 per sample

    regression:
        - target_cols = one or more float columns
        - y is float32 vector per sample
    """

    def __init__(
                self,
                df: pd.DataFrame,
                feature_cols: list[str],
                target_cols: list[str],
                problem_type: str = "classification",
                num_classes: int = 1,
                ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        if len(feature_cols) == 0:
            raise ValueError("feature_cols must be non-empty.")
        if len(target_cols) == 0:
            raise ValueError("target_cols must be non-empty.")

        self.problem_type = problem_type
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.num_classes = num_classes

        # ---- feature matrix ----
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")

        self.X = df[feature_cols].to_numpy(dtype=np.float32)

        # ---- targets ----
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing target columns: {missing}")

        if problem_type == "classification":
            if len(target_cols) != 1:
                raise ValueError(
                    "classification mode requires exactly ONE integer-coded target column."
                )
            y = df[target_cols[0]].astype(np.int64).to_numpy()
        else:  # regression
            y = df[target_cols].to_numpy(dtype=np.float32)

        self.y = y

        # ---- weights ----
        if "Data_weight" in df.columns:
            self.w = df["Data_weight"].to_numpy(dtype=np.float32)
        else:
            self.w = np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        # scalar index for classification
        if self.problem_type == "classification":
            y = torch.tensor(int(self.y[idx]), dtype=torch.int64)
        else:
            y = torch.tensor(self.y[idx], dtype=torch.float32)

        w = torch.tensor(self.w[idx], dtype=torch.float32)
        return x, y, w

class EarlyStopper:
    """
    Stop only when BOTH:
      • No improvement for `patience` epochs, AND
      • Loss variation over last `flat_patience` ≤ flat_delta (or ≤ rel_flat * |best|)

    Optional guard:
      • max_bad_epochs: hard cap of consecutive non-improving epochs

    Notes:
      flat_delta=None disables flat-check entirely.
    """

    def __init__(self,
                 min_delta: float = 0.0,
                 patience: int = 15,
                 burn_in: int = 10,
                 # flat-window
                 flat_delta: float | None = None,
                 flat_patience: int | None = None,
                 flat_mode: str = "iqr",
                 rel_flat: float | None = 2e-3,   # fraction of |best|, ignored if flat_delta=None
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

    # ---------------------------------------------------
    def _flat_metric(self, arr):
        if self.flat_mode == "iqr":
            q75, q25 = np.percentile(arr, [75, 25])
            return float(q75 - q25)
        return float(np.max(arr) - np.min(arr))

    # ---------------------------------------------------
    def step(self, val_loss: float, model=None):
        """Return True if should stop."""
        self.epoch += 1
        self.window.append(val_loss)

        # Improvement?
        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = val_loss
            self.epochs_since_best = 0

            if model is not None:
                # store CPU copy of best weights
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.epochs_since_best += 1

        # Burn-in: never stop before this
        if self.epoch < self.burn_in:
            return False

        # Hard guardrail
        if self.max_bad_epochs is not None and \
           self.epochs_since_best >= self.max_bad_epochs:
            self.stop_reason = (
                f"Reached max_bad_epochs "
                f"({self.epochs_since_best} ≥ {self.max_bad_epochs})"
            )
            return True

        # Need patience before checking flatness
        if self.epochs_since_best < self.patience:
            return False

        # If user disabled flat-window checking
        if self.flat_delta is None:
            self.stop_reason = (
                f"No improvement for patience={self.patience} epochs"
            )
            return True

        # Need full window
        if len(self.window) < self.window.maxlen:
            return False

        # Compute threshold
        thr = self.flat_delta
        if self.rel_flat is not None and np.isfinite(self.best):
            thr = max(thr, self.rel_flat * max(1e-12, abs(self.best)))

        fluct = self._flat_metric(list(self.window))

        if fluct <= thr:
            self.stop_reason = (
                f"AND stop: no-improve {self.epochs_since_best} ≥ {self.patience}, "
                f"flat-window {self.flat_mode}={fluct:.3e} ≤ {thr:.3e}"
            )
            return True

        return False

    # ---------------------------------------------------
    def get_best_state(self):
        """Return a dict of best weights (CPU)."""
        return self.best_state


class MLP(GeneratorFailureProbabilityInference):
    """
    Multi-Layer Perceptron surrogate (PyTorch).

    Two main modes:
    - classification  (default): multi-class with CrossEntropyLoss
        * num_classes > 1
        * target_cols must contain exactly ONE integer-coded column (e.g. Final_gen_state in {0,1,2})
        * output dim = num_classes
        * predict() returns class probabilities (softmax over logits)

    - regression:
        * num_classes should be 1
        * output dim = len(target_cols)
        * predict() returns raw outputs (optionally inverse-transformed if scaler_target is set)
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.model = None
        self.val_loss: list[float] = []

        self.problem_type: str = ""  # set in build_model
        self.num_classes: int = 1

        self.pytorch_activation_functions = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'softmax': nn.Softmax,
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
        num_classes: int=None,
        out_act_fn: Optional[str] = None,
        problem_type: Literal["classification", "regression"] = "classification",
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
        self.num_classes = int(num_classes)
        self.hidden_sizes = hidden_sizes
        self.activations = activations
        self.problem_type = problem_type

        if len(hidden_sizes) != len(activations):
            raise ValueError("hidden_sizes and activations must have the same length.")

        if problem_type == "classification":
            if self.num_classes <= 1:
                raise ValueError("classification mode requires num_classes > 1.")
            if len(self.target_cols) != 1:
                raise ValueError(
                    "classification mode with CrossEntropy requires exactly one "
                    "integer-coded target column (e.g. ['Final_gen_state'])."
                )
            # logits per class
            out_dim = self.num_classes
            if out_act_fn is not None:
                raise ValueError(
                    "Do not set out_act_fn when using classification/CrossEntropy "
                    "(CrossEntropyLoss expects raw logits)."
                )
        elif problem_type == "regression":
            if self.num_classes != 1:
                raise ValueError("regression mode expects num_classes == 1.")
            out_dim = len(self.target_cols)
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

        in_dim = len(self.feature_cols)

        model = nn.Sequential()
        last_dim = in_dim
        for l, (h, act_name) in enumerate(zip(hidden_sizes, activations)):
            if act_name not in self.pytorch_activation_functions:
                raise KeyError(f"Unknown activation '{act_name}'.")
            model.add_module(f"linear_{l}", nn.Linear(last_dim, h))
            model.add_module(f"activation_{l}", self.pytorch_activation_functions[act_name]())
            last_dim = h

        model.add_module('linear_out', nn.Linear(last_dim, out_dim))
        #  output activation only for regression/BCE-style tasks
        if out_act_fn is not None:
            if out_act_fn not in self.pytorch_activation_functions:
                raise KeyError(f"Unknown out_act_fn '{out_act_fn}'.")
            if problem_type == "classification":
                raise ValueError(
                    "out_act_fn must be None for classification with cross_entropy "
                    "(we need raw logits)."
                )
            model.add_module("out_activation", self.pytorch_activation_functions[out_act_fn]())

        self.model = model
        self._init_weights_kaiming()
        

        # record rebuild spec
        self._build_spec = {
            "builder": "build_model",
            "kwargs": {
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "num_classes": num_classes,
                "hidden_sizes": hidden_sizes,
                "activations": activations,
                "out_act_fn": out_act_fn,
                "problem_type": problem_type,
            },
        }

        self.num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.verbose:
            print(self.model)
            print(
                f"Input dim: {in_dim} | Output dim: {out_dim} | "
                f"Trainable params: {self.num_parameters:,}"
            )
    
    def _init_weights_kaiming(self):
        """Apply Kaiming initialization to all Linear layers."""

        for module in self.model.modules():   # assuming self.model is nn.Sequential
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.0,                  # slope for Relu; use 0.01 if LeakyRelu
                    mode='fan_in',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _make_loss(self, loss_name: str):
        """
        Return a PyTorch loss with reduction='none', so we always get
        per-sample losses and can handle weighting ourselves.

        Supported:
        - 'mse', 'mae'           : regression
        - 'logloss'              : binary classification (BCE-with-logits)
        - 'cross_entropy'        : multi-class classification (logits)
        """
        loss_name = loss_name.lower()
        if loss_name == "mse":
            return nn.MSELoss(reduction="none")
        if loss_name == "mae":
            return nn.L1Loss(reduction="none")
        if loss_name == "logloss":
            return nn.BCEWithLogitsLoss(reduction='none')  # expects logits, applies sigmoid internally
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss(reduction="none")  # logits + class indices
        if loss_name == "focal_loss":
            return focal_loss
        raise ValueError(f"Unknown loss '{loss_name}'")
    
    def train_model(
        self,
        optimizer: Literal["adam", "sgd", "rmsprop"] = "adam",
        loss: Literal["mse", "mae", "logloss", "cross_entropy", "focal_loss"] = "cross_entropy",
        focal_loss_gamma: float | None = None,
        focal_loss_gamma_schedule: Optional[str] = None,
        focal_loss_alpha: np.ndarray | None = None,
        focal_loss_alpha_schedule: Optional[str] = None,
        regularization_type: Optional[str] = "L2",
        lambda_reg: float | list[float] = 1e-3,
        epochs: int = 200,
        batch_size: int = 200,
        lr: float = 1e-3,
        weights_data: bool = False,
        device: str = "cpu",
        # smart early stopping
        early_stopping: bool = True,
        patience: int = 20,
        min_delta: float = 0.0,
        flat_delta: float | None = None,
        flat_patience: int | None = None,
        flat_mode: str = "range",  # or "iqr"
        rel_flat: float | None = 2e-3,
        burn_in: int = 10,
        # stability & scheduling
        grad_clip_norm: Optional[float] = None,
        lr_scheduler: Optional[Literal["plateau", "cosine", "onecycle"]] = None,
        scheduler_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Train the MLP with optional data weights, smart early stopping, grad clipping,
        and LR scheduling.

        Parameters
        ----------
        optimizer : {'adam','sgd','rmsprop'}
        loss : {'mse','mae','logloss','cross_entropy'}
            For classification (problem_type='classification'), must be 'cross_entropy'.
        regularization_type : {'L1','L2',None}
        lambda_reg : float | list[float]
            Scalar or per-epoch strengths (len == epochs) if regularization_type is not None.
        epochs : int
        batch_size : int
        lr : float
        device : str
            'cpu' or 'cuda'.
        """

        # --- consistency checks between problem_type and loss ---
        if self.problem_type == "classification" and loss not in ["cross_entropy", "focal_loss", "logloss"]:
            raise ValueError(
                "For classification mode, you must use loss='cross_entropy' "
                "(targets are integer class indices)."
            )
        if self.problem_type == "regression" and loss == "cross_entropy":
            raise ValueError(
                "cross_entropy loss is only valid for classification mode."
            )

        # --- regularization schedule ---
        if regularization_type is not None and not (
            (isinstance(lambda_reg, (float, int)) and float(lambda_reg) > 0.0)
            or (isinstance(lambda_reg, list) and len(lambda_reg) > 0)
        ):
            raise ValueError(
                "With regularization, lambda_reg must be > 0 (float) or a non-empty list."
            )

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
        val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
        loss_fn = self._make_loss(loss)
        optim = self.pytorch_optimizers[optimizer](self.model.parameters(), lr=lr)

        # --- LR scheduler (optional) ---
        scheduler = None
        scheduler_kwargs = scheduler_kwargs or {}
        if lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode="min",
                factor=scheduler_kwargs.get("factor", 0.5),
                patience=scheduler_kwargs.get("patience", 5),
                cooldown=scheduler_kwargs.get("cooldown", 0),
                min_lr=scheduler_kwargs.get("min_lr", 1e-6),
            )
        elif lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=max(1, epochs)
            )
        elif lr_scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=scheduler_kwargs.get("start_factor", 1.0),
                end_factor=scheduler_kwargs.get("end_factor", 0.0),
                total_iters=max(1, epochs)
            )
        elif lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optim,
                gamma=scheduler_kwargs.get("gamma", 0.9)
            )
        elif lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim,
                max_lr=lr,
                steps_per_epoch=max(1, len(train_loader)),
                epochs=epochs,
            )

        # ---- focal loss schedule ----
        if loss.lower() == 'focal_loss' and self.problem_type == 'classification':
            if focal_loss_gamma is None:
                raise ValueError("focal_loss_gamma must be provided for focal_loss.")
            if focal_loss_alpha is None:
                focal_loss_alpha = 1.0 / self.num_classes
            
            if focal_loss_gamma_schedule == 'constant' or focal_loss_gamma_schedule is None or focal_loss_gamma==0.0:
                gammas_focal_ = torch.ones(epochs, device=device).unsqueeze(1) * focal_loss_gamma
            elif focal_loss_gamma_schedule == 'linear':
                gammas_focal_ = torch.linspace(focal_loss_gamma, 0.0, epochs, device=device).unsqueeze(1)
            elif focal_loss_gamma_schedule == 'exponential':
                gammas_focal_ = torch.logspace(np.log10(focal_loss_gamma), np.log10(1e-2), epochs, device=device).unsqueeze(1)
            elif focal_loss_gamma_schedule == 'cosine':
                gammas_focal_ = (focal_loss_gamma / 2) * (1 + torch.cos(torch.linspace(0, np.pi, epochs, device=device))).unsqueeze(1)
            else:
                raise ValueError(f"Unknown focal_loss_gamma_schedule '{focal_loss_gamma_schedule}'.")
            
            if focal_loss_alpha_schedule == 'constant' or focal_loss_alpha_schedule is None:
                alphas_focal_ = torch.tensor(np.array([np.ones(epochs)*a for a in focal_loss_alpha]).T, device=device)
            elif focal_loss_alpha_schedule == 'linear':
                alphas_focal_ = torch.tensor(np.array([np.linspace(a, 1.0, epochs) for a in focal_loss_alpha]).T, device=device)
            elif focal_loss_alpha_schedule == 'exponential':
                alphas_focal_ = torch.tensor(np.array([np.logspace(np.log10(a), np.log10(1), epochs) for a in focal_loss_alpha]).T, device=device)
            elif focal_loss_alpha_schedule == 'cosine':
                alphas_focal_ = torch.tensor(np.array([((a-1) / 2) * (1 + np.cos(np.linspace(0, np.pi, epochs)))+1 for a in focal_loss_alpha]).T, device=device)
            else:
                raise ValueError(f"Unknown focal_loss_alpha_schedule '{focal_loss_alpha_schedule}'.")
            
            if torch.isnan(gammas_focal_).any():
                raise ValueError("NaN detected in focal loss gamma schedule.")
            if torch.isnan(alphas_focal_).any():
                raise ValueError("NaN detected in focal loss alpha schedule.")
            if not torch.isfinite(gammas_focal_).all():
                raise ValueError("Inf detected in focal loss gamma schedule.")
            if not torch.isfinite(alphas_focal_).all():
                raise ValueError("Inf detected in focal loss alpha schedule.")

            # Clamp to safe ranges
            if (gammas_focal_ < 0.0).any():
                print("Warning: negative gammas detected; clamping to 0.0")
            gammas_focal_ = torch.clamp(gammas_focal_, min=0.0)

            if (alphas_focal_ < 0.0).any():
                print("Warning: negative alphas detected; clamping to 0.0")
            alphas_focal_ = torch.clamp(alphas_focal_, min=0.0)




        # --- weighted reduction helper ---
        def _reduce_elemwise_loss(tensor: torch.Tensor, w: Optional[torch.Tensor]):
            """
            Convert elementwise loss to a scalar:

            - tensor: [B] or [B, ...]
            - w: per-sample weights of shape [B] or None

            1) If tensor has extra dims, mean over them -> [B]
            2) If weights are provided, do weighted mean over batch
            else, simple mean over batch.
            """
            # tensor: [B] or [B, ...] -> per-sample mean over feature dims, then (weighted) mean over batch
            if tensor.ndim > 1:
                per_sample = tensor.view(tensor.size(0), -1).mean(dim=1)  # [B]
            else:
                per_sample = tensor  # [B]

            if w is None:
                return per_sample.mean()

            w = w.to(per_sample.dtype)
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

                    # classification: yb should be [B] of class indices
                    if loss.lower() == 'cross_entropy':
                        if yb.ndim != 1:
                            yb = yb.view(-1)
                        yb = yb.long()

                    yhat = self.model(xb)             # logits or continuous output
                    if loss.lower() == 'focal_loss':
                        if train:
                            # get per-epoch alpha/gamma
                            a = alphas_focal_[epoch-1, :].repeat(yb.size(0), 1)  # shape [B, num_classes]
                            g = gammas_focal_[epoch-1].repeat(yb.size(0))  # shape [B]
                            elem = loss_fn(logits=yhat, targets=yb, alpha=a, gamma=g, reduction='none')
                        else:
                            # evaluation: use cross-entropy
                            a = torch.ones((yb.size(0), self.num_classes), device=device)
                            g = torch.zeros((yb.size(0),), device=device)
                            yhat = true_prob_focal_loss(yhat, gamma=gammas_focal_[epoch-1])
                            elem = loss_fn(probs=yhat, targets=yb, alpha=a, gamma=g, reduction='none')
                    else:
                        elem = loss_fn(yhat, yb)          # elementwise, shape [B] or [B,...]
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
        stopper = EarlyStopper(
            min_delta=min_delta,
            patience=patience,
            burn_in=burn_in,
            flat_delta=flat_delta,
            flat_patience=flat_patience,
            flat_mode=flat_mode,
            rel_flat=rel_flat,
        )

        # --- training loop ---
        for ep in range(1, epochs + 1):
            train_loss = step(train_loader, True, ep)
            val_loss = step(val_loader, False, ep)
            self.val_loss.append(val_loss)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()

            if self.verbose:
                print(
                    f"Epoch {ep:03d}: train={train_loss:.4e} | val={val_loss:.4e}"
                )

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

        For classification:
            - returns class probabilities of shape (N, num_classes).
        For regression:
            - returns raw outputs of shape (N, len(target_cols)), optionally inverse-transformed.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        device = next(self.model.parameters()).device
        X_df = X.copy()

        # standardize selected features
        if self.standardize is True and self.scaler_feature is not None:
            X_df.loc[:, self.feature_cols] = self.scaler_feature.transform(
                X_df[self.feature_cols].to_numpy()
            )
            stand_targets = list(self.target_cols)  # for regression only
        elif isinstance(self.standardize, list) and self.scaler_feature is not None:
            stand_feat = [
                c
                for c in self.feature_cols
                if c in self.standardize and c in X_df.columns
            ]
            X_df.loc[:, stand_feat] = self.scaler_feature.transform(
                X_df[stand_feat].to_numpy()
            )
            stand_targets = [c for c in self.target_cols if c in self.standardize]
        else:
            stand_targets = []

        X_np = X_df[self.feature_cols].to_numpy(dtype=np.float32)
        logits_or_out = (
            self.model(
                torch.tensor(X_np, dtype=torch.float32, device=device)
            )
            .detach()
            .cpu()
            .numpy()
        )

        if self.problem_type == "classification":
            # probabilities over classes
            y_pred = torch.softmax(torch.tensor(logits_or_out), dim=1).numpy()
            return y_pred

        # regression: optionally inverse-transform targets
        y_pred = logits_or_out
        if self.scaler_target is not None and stand_targets:
            y_df = pd.DataFrame(y_pred, columns=self.target_cols)
            y_df.loc[:, stand_targets] = self.scaler_target.inverse_transform(
                y_df[stand_targets].to_numpy()
            )
            y_pred = y_df[self.target_cols].to_numpy(dtype=np.float32)
        return y_pred

    # ---------------------- Save / Load ----------------------

    def save_model(self, model_path: str) -> None:
        """
        Save a self-describing checkpoint that can rebuild the model and restore metadata.
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
                "sklearn": StandardScaler.__module__.split(".")[0],
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
                "scaler_feature": _pickle_or_none(
                    getattr(self, "scaler_feature", None)
                ),
                "scaler_target": _pickle_or_none(
                    getattr(self, "scaler_target", None)
                ),
                "problem_type": getattr(self, "problem_type", "classification"),
                "num_classes": getattr(self, "num_classes", 1),
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
    def load_model(
        cls, model_path: str, map_location: str = "cpu", verbose: bool = True
    ):
        """
        Rebuild an MLP (or subclass) instance from a checkpoint.
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
        obj.problem_type = data_section.get("problem_type", "classification")
        obj.num_classes = data_section.get("num_classes", 1)

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

        obj.scaler_feature = _unpickle_or_none(
            data_section.get("scaler_feature", None)
        )
        obj.scaler_target = _unpickle_or_none(
            data_section.get("scaler_target", None)
        )

        train_section = ckpt.get("train", {})
        obj.optimizer_name = train_section.get("optimizer_name", None)
        obj.loss_fn_name = train_section.get("loss_fn_name", None)
        obj.val_loss = train_section.get("val_loss", [])
        obj.num_parameters = train_section.get("num_parameters", None)

        obj.model.eval()
        if verbose:
            print(f"Loaded {class_name} from {model_path}")
            print(
                f"Rebuilt with {build_spec['builder']}(**{list(build_spec['kwargs'].keys())})"
            )
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
        imp = imp.loc[imp["Feature"].str.startswith('State_') == False]  # exclude State_ features
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
    XGBoost multi-class classifier wrapper (scikit-learn style).

    This wrapper is **classification-only** and assumes:

    - A single integer-encoded target column (e.g. 0..num_classes-1).
    - Features are already standardized in-place by
      `GeneratorFailureProbabilityInference.prepare_data` if requested.
    - The model outputs class probabilities via `predict_proba`.

    Typical usage:
        model = xgboostModel(verbose=True)
        model.build_model(
            max_depth=4,
            eta=0.1,
            gamma=0.0,
            reg_lambda=1.0,
            num_boost_round=200,
            feature_cols=feature_names,
            target_cols=["Final_gen_state"],
            num_classes=3,
        )
        model.prepare_data(df, standardize=False)  # or True / list, handled upstream
        model.train_model(weights_data=True)
        probs = model.predict(test_df[feature_names])
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        # classification-only
        self.model: Optional[xgb.XGBClassifier] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build_model(
                    self,
                    max_depth: int,
                    eta: float,
                    gamma: float,
                    reg_lambda: float,
                    num_boost_round: int = 100,
                    feature_cols: Optional[list[str]] = None,
                    target_cols: Optional[list[str]] = None,
                    num_classes: int = 3,
                    eval_metric: str = "mlogloss",
                    objective: str = "multi:softprob",
                    early_stopping_rounds: int = 10,
                    subsample: float = 1.0,
                    device: str = "cpu",
                    ) -> None:
        """
        Instantiate the XGBClassifier with requested hyperparameters.

        Parameters
        ----------
        max_depth : int
            Maximum tree depth.
        eta : float
            Learning rate.
        gamma : float
            Minimum loss reduction required to split.
        reg_lambda : float
            L2 regularization term on leaf weights.
        num_boost_round : int
            Number of boosting rounds (n_estimators).
        feature_cols : list[str] | None
            Names of feature columns.
        target_cols : list[str] | None
            Single-element list with the target column name.
        num_classes : int
            Total number of classes (manual; may exceed the classes present in this dataset).
        eval_metric : str
            Evaluation metric, e.g. 'mlogloss'.
        objective : str
            XGBoost objective, fixed to a multi-class classification objective.
        early_stopping_rounds : int
            Patience on validation metric.
        subsample : float
            Row subsampling ratio.
        device : str
            'cpu' or GPU device string (requires appropriate XGBoost build).
        """
        self.feature_cols = feature_cols or []
        self.target_cols = target_cols or []
        if len(self.target_cols) != 1:
            raise ValueError(
                f"xgboostModel expects exactly one target column; got {self.target_cols!r}"
            )

        self.num_classes = int(num_classes)
        self.max_depth = int(max_depth)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.reg_lambda = float(reg_lambda)
        self.num_boost_round = int(num_boost_round)
        self.eval_metric = eval_metric
        self.objective = objective
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.subsample = float(subsample)
        self.device = device

        # Classification-only estimator
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            eta=self.eta,
            gamma=self.gamma,
            reg_lambda=self.reg_lambda,
            n_estimators=self.num_boost_round,
            subsample=self.subsample,
            eval_metric=self.eval_metric,
            objective=self.objective,
            early_stopping_rounds=self.early_stopping_rounds,
            num_class=self.num_classes,
            verbosity=1 if self.verbose else 0,
            device=self.device,
        )

        # record rebuild spec for save/load
        self._build_spec = {
            "builder": "build_model",
            "kwargs": {
                "max_depth": self.max_depth,
                "eta": self.eta,
                "gamma": self.gamma,
                "reg_lambda": self.reg_lambda,
                "num_boost_round": self.num_boost_round,
                "feature_cols": self.feature_cols,
                "target_cols": self.target_cols,
                "num_classes": self.num_classes,
                "eval_metric": self.eval_metric,
                "objective": self.objective,
                "early_stopping_rounds": self.early_stopping_rounds,
                "subsample": self.subsample,
                "device": self.device,
            },
        }

        if self.verbose:
            print(self.model)
    
    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train_model(self, weights_data: bool = False) -> None:
        """
        Fit the classifier on train/val splits.

        Parameters
        ----------
        weights_data : bool
            If True, use 'Data_weight' as sample weights for train and val.

        Notes
        -----
        - Assumes `prepare_data` has already been called, and that
          standardization (if any) has been applied in-place to
          `self.train_data` and `self.val_data`.
        - `target_cols` must contain exactly one column with integer labels
          in [0, num_classes-1], even if some classes are absent in this
          particular dataset.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        if not hasattr(self, "train_data") or not hasattr(self, "val_data"):
            raise ValueError("Train/val data not found. Call prepare_data() first.")

        target_col = self.target_cols[0]

        # y must be 1D with integer labels
        y_train = self.train_data[target_col].to_numpy(dtype=np.int64)
        y_val = self.val_data[target_col].to_numpy(dtype=np.int64)

        # If the model has been fitted before with a different label set,
        # reset it to avoid XGBoost's "Invalid classes inferred" error.
        if hasattr(self.model, "classes_"):
            prev = np.asarray(self.model.classes_)
            curr = np.unique(y_train)
            if not np.array_equal(prev, curr):
                if self.verbose:
                    print(
                        "[xgboostModel] Detected label set change "
                        f"{prev.tolist()} -> {curr.tolist()}; resetting estimator."
                    )
                self.reset_model()

        # Sample weights
        if weights_data and "Data_weight" in self.train_data.columns:
            w_train = self.train_data["Data_weight"].to_numpy(dtype=np.float32)
            w_val = self.val_data["Data_weight"].to_numpy(dtype=np.float32)
        else:
            w_train = None
            w_val = None

        X_train = self.train_data[self.feature_cols]
        X_val = self.val_data[self.feature_cols]


        self.model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val] if w_val is not None else None,
        )
    
    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_model(self) -> None:
        """
        Recreate the classifier with original hyperparameters (weights discarded).
        """
        if self.model is not None:
            self.model = xgb.XGBClassifier(
                max_depth=self.max_depth,
                eta=self.eta,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                n_estimators=self.num_boost_round,
                subsample=self.subsample,
                eval_metric=self.eval_metric,
                objective=self.objective,
                early_stopping_rounds=self.early_stopping_rounds,
                num_class=self.num_classes,
                verbosity=1 if self.verbose else 0,
                device=self.device,
            )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for new inputs.

        Parameters
        ----------
        X : pd.DataFrame
            Input features in original scale. If standardization was
            applied in `prepare_data`, the same scaler is used here
            to transform X before prediction.

        Returns
        -------
        y_pred : np.ndarray
            Array of shape (N, num_classes) with class probabilities.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        X_df = X.copy()

        # Feature standardization is controlled by GeneratorFailureProbabilityInference.prepare_data.
        if self.standardize is True and self.scaler_feature is not None:
            # All features were standardized during training
            X_df.loc[:, self.feature_cols] = self.scaler_feature.transform(
                X_df[self.feature_cols].to_numpy()
            )
        elif isinstance(self.standardize, list) and self.scaler_feature is not None:
            # Only some features were standardized
            stand_feat = [
                c for c in self.feature_cols
                if c in self.standardize and c in X_df.columns
            ]
            if stand_feat:
                X_df.loc[:, stand_feat] = self.scaler_feature.transform(
                    X_df[stand_feat].to_numpy()
                )

        X_np = X_df[self.feature_cols].to_numpy(dtype=np.float32)
        y_pred = self.model.predict_proba(X_np)

        # Ensure 2D
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save_model(self, model_path: str) -> None:
        """
        Save a single-file checkpoint with XGBoost booster + metadata.
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
                "sklearn": StandardScaler.__module__.split(".")[0],
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
                "scaler_feature": _pickle_or_none(
                    getattr(self, "scaler_feature", None)
                ),
                # kept for compatibility, but unused for classification
                "scaler_target": _pickle_or_none(
                    getattr(self, "scaler_target", None)
                ),
            },
            "train": {
                "early_stopping_rounds": getattr(
                    self, "early_stopping_rounds", None
                ),
                "num_classes": getattr(self, "num_classes", None),
            },
        }

        torch.save(checkpoint, model_path)
        if self.verbose:
            print(f"Saved XGBoost checkpoint to: {model_path}")

    @classmethod
    def load_model(cls, model_path: str, verbose: bool = True):
        """
        Reconstruct an xgboostModel instance from a checkpoint.
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
        obj.num_classes = train_section.get(
            "num_classes", getattr(obj, "num_classes", None)
        )

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

            # restore sklearn wrapper metadata
            try:
                n_classes = getattr(obj, "num_classes", None)
                if n_classes is None or n_classes < 1:
                    n_classes = sk_params.get("num_class", None)
                if n_classes is not None:
                    obj.model.n_classes_ = int(n_classes)
                    obj.model.classes_ = np.arange(int(n_classes))
            except Exception as e:
                if verbose:
                    print(f"[warning] Failed to set n_classes_: {e}")

            try:
                obj.model.n_features_in_ = len(obj.feature_cols or [])
            except Exception:
                pass

        def _unpickle_or_none(b):
            try:
                return pickle.loads(b) if b is not None else None
            except Exception:
                return None

        obj.scaler_feature = _unpickle_or_none(
            data_section.get("scaler_feature", None)
        )
        obj.scaler_target = _unpickle_or_none(
            data_section.get("scaler_target", None)
        )

        obj.early_stopping_rounds = train_section.get(
            "early_stopping_rounds", None
        )

        if verbose:
            print(f"Loaded {class_name} from {model_path}")
            print(
                f"Rebuilt with {build_spec['builder']}(**{list(build_spec['kwargs'].keys())})"
            )
            if booster_raw is None:
                print("Note: booster was not saved (model likely not fitted yet).")
        return obj
    

    def get_feature_importance(
                    self, 
                    importance_type: str = "weight"
                    ) -> dict[str, float]:
        """
        Return feature importance from the fitted booster, mapped to feature names.

        Parameters
        ----------
        importance_type : str
            One of {'weight', 'gain', 'cover', 'total_gain', 'total_cover'}.

        Returns
        -------
        importance : dict[str, float]
            Mapping from feature name -> importance score.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model/fit first.")
        booster = self.model.get_booster()
        raw_imp = booster.get_score(importance_type=importance_type)  # keys like 'f0','f1',...
        mapped: dict[str, float] = {}
        for k, v in raw_imp.items():
            # Map 'f0' -> feature_cols[0] where possible
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(self.feature_cols):
                    name = self.feature_cols[idx]
                else:
                    name = k
            else:
                name = k
            mapped[name] = float(v)
        return mapped

    def plotFeatureImportance(
                        self,
                        importance_criterions: list[str] = ("weight", "gain", "cover"),
                        n_features: int = 10,
                        exclude_states: bool = True,
                        ) -> None:
        """
        Plot the top-N features for several importance criteria.

        Parameters
        ----------
        importance_criterions : list[str]
            Criteria to display, e.g. ['weight','gain','cover'].
        n_features : int
            Top features to show per criterion.
        exclude_states : bool
            If True, drop features whose name contains 'State'.
        """
        fig, axs = plt.subplots(
            len(importance_criterions),
            1,
            figsize=(max(6, 0.7 * n_features), 4 * len(importance_criterions)),
        )
        axs = np.atleast_1d(axs)

        for j, criterion in enumerate(importance_criterions):
            importance = self.get_feature_importance(importance_type=criterion)
            imp_df = (
                pd.DataFrame(
                    list(importance.items()), columns=["Feature", "Importance"]
                )
                .sort_values(by="Importance", ascending=False)
                .reset_index(drop=True)
            )
            if exclude_states:
                imp_df = imp_df[~imp_df["Feature"].str.contains("State")]

            top = imp_df.head(n_features)
            axs[j].bar(top["Feature"], top["Importance"])
            axs[j].set_title(f"Feature Importance – {criterion}", fontsize=14)
            axs[j].set_ylabel("Importance Score", fontsize=12)
            axs[j].tick_params(axis="x", labelrotation=45)

        plt.tight_layout()
        plt.show()
