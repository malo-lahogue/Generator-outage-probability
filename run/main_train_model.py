#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time

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
        description="Training of MLP or XGBoost."
    )

    # Data paths
    p.add_argument("--failures",  type=Path, default=THIS_DIR / "../DATA/hourly/hourly_failure_dataset_compressed.csv")
    p.add_argument("--weather",   type=Path, default=THIS_DIR / "../DATA/hourly/hourly_weather_by_state.csv")
    p.add_argument("--powerload", type=Path, default=THIS_DIR / "../DATA/hourly/hourly_load_by_state.csv")

    # Problem params
    p.add_argument("--technologies", type=str, default="thermal", help="Group of technologies to consider.")
    p.add_argument("--initial_state", type=str, default="A", help="Which initial MC state to filter on.")

    
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
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])
    power =  pd.read_csv(powerload_path, parse_dates=["UTC time"])
    base = list(weather.columns) + list(power.columns) + ['Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend', 'Technology']

    # Remove duplicates but keep stable order
    seen = set()
    base = [c for c in base if not (c in seen or seen.add(c))]
    # Drop known non-features if present
    drop = {'datetime', 'UTC time', 'Datetime_UTC', 'Datetime'}
    feats = [c for c in base if c not in drop]
    feats = list(set([(name[0].upper() + name[1:]) if isinstance(name, str) and name else name for name in feats]))
    feats.sort()
    return feats


def main() -> None:
    """ Main function to train the ML model. """
    t_start = time()

    args = parse_args()         # Arguments
    np.random.seed(args.seed)   # Reproducibility

    # I/O prep
    ensure_inputs_exist(args.failures, args.weather, args.powerload)
    print(f"Computing transition probabilities starting with {args.models} from state {args.initial_state} for {args.technologies} generators.")

    # ---------- Get feature set ----------
    feature_names = load_feature_bases(args.weather, args.powerload)
    print(f"{len(feature_names)} initial features: {feature_names}")

    technologies = {'nuclear': ['Nuclear'],
                    'hydro': ['Pumped Storage/Hydro'],
                    'geothermal': ['Geothermal'],
                    'thermal': ['CC GT units ', 
                                'CC steam units', 
                                'Co-generator Block ', 
                                'CoG GT units', 
                                'CoG steam units ', 
                                'Combined Cycle Block', 
                                'Fluidized Bed', 'Fossil-Steam', 
                                'Gas Turbine/Jet Engine (Simple Cycle Operation)', 
                                'Gas Turbine/Jet Engine with HSRG', 
                                'Internal Combustion/Reciprocating Engines',
                                'Multi-boiler/Multi-turbine']}.get(args.technologies.lower(), None)
    if technologies is None:
        raise ValueError(f"Unknown technology group: {args.technologies}. Choose from 'nuclear', 'hydro', 'geothermal', or 'thermal'.")
    
    test_periods = [(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31'))]

    # ---------- Merge + label prep ----------
    train_val_df, test_df, feature_names, target_columns, integer_encoding = im.preprocess_data(failure_data_path=args.failures,
                                                                                weather_data_path=args.weather,
                                                                                power_load_data_path=args.powerload,
                                                                                feature_names=feature_names,
                                                                                cyclic_features=["Season", "Month", "DayOfWeek", "DayOfYear"],
                                                                                state_one_hot=True,
                                                                                initial_MC_state_filter=args.initial_state,
                                                                                technology_filter=technologies,
                                                                                test_periods=test_periods
                                                                                )

    subset_length = 1000000
    train_val_df = train_val_df.iloc[0:subset_length].copy().reset_index(drop=True)
    print(f"Train/Val Dataset shape: {train_val_df.shape}")

    
    # Standardize all continuous features (exclude one-hots and raw categorical/cyclic markers)
    exclude = {"Holiday", "Weekend", "Season", "Month", "DayOfWeek", "DayOfYear"}
    stand_cols = [f for f in feature_names if not f.startswith("State_") and not f.endswith("_isnan") and not f.endswith("_sin") and not f.endswith("_cos") and f not in exclude]
    print(f"Standardized features ({len(stand_cols)}): {stand_cols}")
    

    if "xgb" in args.models:

        xgb_model = im.xgboostModel(verbose=False)
        xgb_model.build_model(max_depth=6,
                            eta=0.07,
                            gamma=0.8,
                            reg_lambda=1,
                            num_boost_round=100,
                            feature_cols=feature_names,
                            target_cols=target_columns,
                            num_classes=3,
                            eval_metric='mlogloss', # rmse, logloss, mae, mape
                            objective='multi:softprob',
                            subsample=1,
                            device=args.device,
                            early_stopping_rounds=10)
        

        xgb_model.prepare_data(train_val_df, train_ratio=0.80, val_ratio=0.2, standardize=stand_cols)

        xgb_model.train_model(weights_data=True)

        path_to_save = THIS_DIR / "../Results/Models" / f"XGB_global_model_{args.technologies}_{args.initial_state}.pth"

        xgb_model.save_model(path_to_save)


    elif "mlp" in args.models:


        mlp_model = im.MLP(verbose=False)
        mlp_model.build_model(feature_cols=feature_names,
                            target_cols=target_columns,
                            num_classes=3,
                            hidden_sizes=(100, 50),
                            activations=('relu', 'relu'))

        mlp_model.prepare_data(train_val_df, train_ratio=0.80, val_ratio=0.2, standardize=stand_cols)
        mlp_model.train_model(optimizer='adam', 
                              loss='cross_entropy',
                                regularization_type='L2', lambda_reg=1e-3,
                                weights_data=True,
                                epochs=10, batch_size=1024, lr=2e-4,
                                device=args.device,
                                # smart stopping knobs
                                early_stopping=True, patience=15, min_delta=1e-4,
                                flat_delta=1e-3, flat_patience=20, flat_mode='iqr', rel_flat=2e-3, burn_in=10,
                                # stability & LR policy
                                grad_clip_norm=1.0,
                                lr_scheduler='plateau', scheduler_kwargs={'factor':0.5, 'patience':3, 'min_lr':1e-6})

        path_to_save = THIS_DIR / "../Results/Models" / f"MLP_model_{args.technologies}_{args.initial_state}.pth"
        mlp_model.save_model(path_to_save)

    t_end = time()
    print(f"Training completed in {t_end - t_start:.1f} seconds.")




if __name__ == "__main__":
    main()