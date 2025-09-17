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

    # Grid search params
    p.add_argument("--result_csv",    type=Path, default=THIS_DIR / "../Results/grid_search_log_per_state_XGB_new.csv")
    p.add_argument("--top_keep",      type=percent01, default=0.33, help="Fraction kept at each halving level.")
    p.add_argument("--val_frac",      type=percent01, default=0.20, help="Validation fraction.")
    p.add_argument("--reuse_results", default=True, help="Reuse rows already computed in result.")

    # Runtime / reproducibility
    p.add_argument("--seed",   type=int, default=123, help="Random seed for splits / reproducibility.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="Default device to request in model specs.")
    p.add_argument("--models", type=str, nargs="+", default=["both"], choices=["xgb", "mlp", "both"],
                   help='Which models to include: "xgb", "mlp", or "both".')

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
    args.result_csv.parent.mkdir(parents=True, exist_ok=True)

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
        feature_na_drop_threshold=0.10,
        sort_by_date=True,
    )

    # Standardize all continuous features (exclude one-hots and raw categorical/cyclic markers)
    exclude = {"Holiday", "Weekend", "Season", "Month", "DayOfWeek",  "DayOfYear", "State"}
    stand_cols = [f for f in feature_cols if  all([not f.startswith(exc) for exc in exclude])]
    print(f"Standardized features ({len(stand_cols)}): {stand_cols}")
    

    # ---------- Define model specs ----------
    # For both models, we define:
    # - common_build: parameters used in all builds
    # - build_grid:   parameters to search at build time
    # - common_train: parameters used in all training runs
    # - train_grid:   parameters to search at training time

    # 1) XGBoost
    xgb_common_build = {
                        "feature_cols" : feature_cols,
                        "target_cols"  : target_cols,
                        "eval_metric"  : "logloss",
                        "objective"    : "reg:logistic",
                        "early_stopping_rounds" : 10,
                        "subsample"       : 1.0,
                        # "num_boost_round" : 500,
                        "device"          : args.device,     
                        }
    xgb_build_grid = {
                        "max_depth":   [2, 4, 6, 8, 10, 12],
                        "eta":         [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1],
                        "gamma":       [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],# [0.0, 0.25, 0.5, 0.75, 1.0],
                        "reg_lambda":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],#[0.0, 0.25, 0.5, 0.75, 1.0]
                        #
                        "num_boost_round" : [1200],# [100, 200, 500, 800, 1200, 1500]
                        }

    xgb_common_train = {
                        "weights_data": True,
                        # "seed": args.seed,
                     }

    # 2) MLP
    mlp_common_build = {
        "feature_cols": feature_cols,
        "target_cols":  target_cols,
        "out_act_fn":   "sigmoid",
    }
    mlp_build_grid = {
        "hidden_sizes": [
                        # (128, 128, 64),#
                        # (256, 128, 64),#
                        # (256, 256, 128, 64),#
                        # (256, 256, 256, 128, 64),
                        # (256, 256, 128, 64,  32, 16, 8, 4),#
                        # (1024, 256, 64),
                        # (2048, 512, 64),
                        # (2048, 512),
                        (8192, 1024, 64),#
                        ],
        "activations": [
                        # ("relu",) * 3,
                        # ("relu",) * 3,
                        # ("relu",) * 4,
                        # ("relu",) * 5,
                        # ("relu",) * 8,
                        # ("relu",) * 3,
                        # ("relu",) * 3,
                        # ("relu",) * 2,
                        ("relu",) * 3,
                        ],
    }
    mlp_common_train =  {
                        "optimizer": "adam",
                        "loss": "logloss",
                        "regularization_type": "L2",
                        "weights_data": True,
                        "device": args.device,
                        "early_stopping": True,
                        "grad_clip_norm": 1.0,
                        "lr_scheduler": "plateau",
                        # "seed": args.seed,
                        }
    
    mlp_train_grid = {
                    "lambda_reg"          : [5e-5, 1e-4, 2e-4, 4e-4, 1e-3],
                    "epochs"              : [2000],   # upper bound — levels will cap
                    "batch_size"          : [128, 256],
                    "lr"                  : [1e-4, 2e-4, 4e-4, 1e-3],
                    "patience"            : [50],
                    "min_delta"           : [5e-5],
                    "flat_delta"          : [1e-4],
                    "flat_patience"       : [50],
                    "flat_mode"           : ['iqr'],
                    "rel_flat"            : [2e-3],
                    "burn_in"             : [150],
                    }
    #                 Stop only when BOTH conditions hold:
    #                   • No-improve: best hasn't improved by min_delta for `patience` epochs, AND
    #                   • Flat-window: variability in last `flat_patience` epochs <= flat_delta (abs or relative).


    # Choose models from CLI
    include_xgb = "xgb" in args.models or "both" in args.models
    include_mlp = "mlp" in args.models or "both" in args.models

    model_specs = []
    if include_xgb:
        model_specs.append({
            "name":         "xgboostModel",
            "constructor":  lambda: im.xgboostModel(verbose=False),
            "common_build": xgb_common_build,
            "build_grid":   xgb_build_grid,
            "common_train": xgb_common_train,
            "train_grid":   {},
        })
    if include_mlp:
        model_specs.append({
            "name":         "MLP",
            "constructor":  lambda: im.MLP(verbose=False),
            "common_build": mlp_common_build,
            "build_grid":   mlp_build_grid,
            "common_train": mlp_common_train,
            "train_grid":   mlp_train_grid,
        })

    if not model_specs:
        raise SystemExit("No models selected. Use --models xgb|mlp|both")

    # Validation metric per model
    val_metric = {spec["name"]: "logloss" for spec in model_specs}

    # Training levels (successive halving caps)
    n_rows = len(data_df)
    training_levels = [
        {"name": "L1-fast",   "epochs": 250,  "data_cap": int(n_rows * 0.60)},
        # {"name": "L2-medium", "epochs": 500,  "data_cap": int(n_rows * 0.80)},
        # {"name": "L3-full",   "epochs": 2000, "data_cap": None},
    ]

    # ---------- Run search ----------
    winners = im.successive_halving_search(
        model_specs=model_specs,
        data=data_df,
        standardize=stand_cols,
        result_csv=str(args.result_csv),
        train_ratio=1.0 - args.val_frac,
        val_ratio=args.val_frac,
        val_metric_per_model=val_metric,
        levels=training_levels,
        top_keep_ratio=args.top_keep,
        resume=args.reuse_results,
        # seed=args.seed,  # if supported by your helper
    )


    # ---------- Summarize results ----------
    print("\n=== Global winner ===")

    best_model = min(winners, key=lambda x: x[3])
    mi, build_p, train_p, score = best_model
    spec = model_specs[mi]
    print(f"[{spec['name']}] score={score:.6f}")
    print(" build:", json.dumps(build_p, sort_keys=True))
    print(" train:", json.dumps(train_p, sort_keys=True))
    print("-" * 60)


if __name__ == "__main__":
    main()