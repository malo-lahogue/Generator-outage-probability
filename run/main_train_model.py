#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local src/ is importable
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((THIS_DIR / "../src").resolve()))
import inferenceModels as im


# --------------------------- CLI ---------------------------

def percent01(x: str) -> float:
    """ Argparse type: float in (0, 1) """
    v = float(x)
    if not (0.0 < v < 1.0):
        raise argparse.ArgumentTypeError("must be in (0, 1)")
    return v

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Successive-halving grid search for MLP & XGBoost."
    )

    # Data paths
    p.add_argument("--failures",  type=Path, default=THIS_DIR / "../DATA/filtered_events.csv")
    p.add_argument("--events",    type=Path, default=THIS_DIR / "../DATA/event_count.csv")
    p.add_argument("--weather",   type=Path, default=THIS_DIR / "../DATA/weather_data_per_state_all.csv")
    p.add_argument("--powerload", type=Path, default=THIS_DIR / "../DATA/power_load_input.csv")

    # Problem params
    p.add_argument("--target",   type=str, choices=["Unit_Failure", "Frequency"], default="Frequency",
                   help='Prediction target. "Unit_Failure" = unit-level labels, "Frequency" = daily-state aggregate.')
    p.add_argument("--clusters", type=int, default=1, help="Cause-code clusters (1 = no clustering).")

    
    # Runtime / reproducibility
    p.add_argument("--seed",   type=int, default=123, help="Random seed for splits / reproducibility.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Default device to request in model specs.")
    p.add_argument("--models", type=str, nargs="+", default=["both"], choices=["xgb", "mlp"],
                   help='Which models to include: "xgb" or "mlp".')

    return p.parse_args()


# --------------------------- Helpers ---------------------------

def ensure_inputs_exist(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n  " + "\n  ".join(missing))

def load_feature_bases(weather_path: Path, powerload_path: Path) -> list[str]:
    weather = pd.read_csv(
        weather_path, index_col=["Date", "State"], parse_dates=["Date"],
        usecols=lambda c: c not in ["Unnamed: 0"]
    )
    power = pd.read_csv(
        powerload_path, index_col=["Date", "State"], parse_dates=["Date"],
        usecols=lambda c: c not in ["Unnamed: 0"]
    )
    base = list(weather.columns) + list(power.columns) + ["Season", "Month", "DayOfWeek", "DayOfYear", "Holiday", "Weekend"]
    # Remove duplicates but keep stable order
    seen = set()
    base = [c for c in base if not (c in seen or seen.add(c))]
    # Drop known non-features if present
    drop = {"EventStartDT", "Date", "PRCP_30dz"}
    feats = [c for c in base if c not in drop]
    feats.sort()
    return feats


def main() -> None:
    """ Main function to run the grid search. """
    args = parse_args()         # Arguments
    np.random.seed(args.seed)   # Reproducibility

    # I/O prep
    ensure_inputs_exist(args.failures, args.events, args.weather, args.powerload)

    # ---------- Get feature set ----------
    feature_names = load_feature_bases(args.weather, args.powerload)
    print(f"{len(feature_names)} initial features: {feature_names}")

    # ---------- Merge + label prep ----------
    data_df, feature_cols, target_cols = im.preprocess_data(
        failure_path=args.failures,
        event_count_path=args.events,
        weather_data_path=args.weather,
        power_data_path=args.powerload,
        feature_names=feature_names,
        target=args.target,
        state_one_hot=True,
        cyclic_features=["Season", "Month", "DayOfWeek", "DayOfYear"],
        cause_code_n_clusters=args.clusters,
        feature_na_drop_threshold=0.10
    )

    # Standardize all continuous features (exclude one-hots and raw categorical/cyclic markers)
    exclude = {"Holiday", "Weekend", "Season", "Month", "DayOfWeek", "DayOfYear"}
    stand_cols = [f for f in feature_cols if not f.startswith("State_") and f not in exclude]
    print(f"Standardized features ({len(stand_cols)}): {stand_cols}")
    

    if "xgb" in args.models:
        xgb_model = im.xgboostModel(verbose=True)
        xgb_model.build_model(max_depth=8,
                            eta=0.02,
                            gamma=1,
                            reg_lambda=1,
                            num_boost_round=500,
                            feature_cols=feature_cols,
                            target_cols=target_cols,
                            eval_metric='logloss', # rmse, logloss, mae, mape
                            objective='reg:logistic',
                            subsample=1)

        xgb_model.prepare_data(data_df, train_ratio=0.80, val_ratio=0.1, test_ratio=0.1, standardize=stand_cols)

        xgb_model.train_model(weights_data=True)

        path_to_save = THIS_DIR / "../Results/Models" / "XGBoost_model.pth"

        xgb_model.save_model(path_to_save)


    elif "mlp" in args.models:
        pass

if __name__ == "__main__":
    main()