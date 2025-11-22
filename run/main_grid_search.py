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
import preprocess_data as ppd
import grid_search as gs


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
    p.add_argument("--failures",  type=Path, default=THIS_DIR / "../DATA/hourly/hourly_failure_dataset_compressed.csv")
    p.add_argument("--weather",   type=Path, default=THIS_DIR / "../DATA/hourly/hourly_weather_by_state.csv")
    p.add_argument("--powerload", type=Path, default=THIS_DIR / "../DATA/hourly/hourly_load_by_state.csv")

    # Problem params
    p.add_argument("--technologies", type=str, default="thermal", help="Group of technologies to consider.")
    p.add_argument("--initial_state", type=str, default="A", help="Which initial MC state to filter on.")
    p.add_argument("--final_state", type=str, default="all", help="Which Final MC state to target.")
    p.add_argument("--states", type=str, default="all", help="States (geographical) to filter on.")


    # Grid search params
    p.add_argument("--result_csv",    type=Path, default=THIS_DIR / "../Results/grid_search_log")# .csv added later
    p.add_argument("--top_keep",      type=percent01, default=0.33, help="Fraction kept at each halving level.")
    p.add_argument("--val_frac",      type=percent01, default=0.20, help="Validation fraction.")
    p.add_argument("--reuse_results", default=True, help="Reuse rows already computed in result.")

    # Runtime / reproducibility
    p.add_argument("--seed",   type=int, default=42, help="Random seed for splits / reproducibility.")
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
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])
    power =  pd.read_csv(powerload_path, parse_dates=["UTC time"])
    base = list(weather.columns) + list(power.columns) + ['Season', 'Month', 'DayOfWeek', 'DayOfYear', 'Holiday', 'Weekend']#, 'Technology']

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
    """ Main function to run the grid search. """
    args = parse_args()         # Arguments
    np.random.seed(args.seed)   # Reproducibility

    # I/O prep
    ensure_inputs_exist(args.failures, args.weather, args.powerload)
    args.states = args.states.replace('_', ' ')
    print(f"Grid search for transition probabilities starting with {args.models} from state {args.initial_state} for {args.technologies} generators in state {args.states}.")

    # ---------- Get feature set ----------
    feature_names = load_feature_bases(args.weather, args.powerload)
    print(f"{len(feature_names)} initial features: {feature_names}")

    # technologies = {'nuclear': ['Nuclear'],
    #                 'hydro': ['Pumped Storage/Hydro'],
    #                 'geothermal': ['Geothermal'],
    #                 'thermal': ['CC GT units ', 
    #                             'CC steam units', 
    #                             'Co-generator Block ', 
    #                             'CoG GT units', 
    #                             'CoG steam units ', 
    #                             'Combined Cycle Block', 
    #                             'Fluidized Bed', 'Fossil-Steam', 
    #                             'Gas Turbine/Jet Engine (Simple Cycle Operation)', 
    #                             'Gas Turbine/Jet Engine with HSRG', 
    #                             'Internal Combustion/Reciprocating Engines',
    #                             'Multi-boiler/Multi-turbine']}.get(args.technologies.lower(), None)
    technologies = ['Gas Turbine/Jet Engine (Simple Cycle Operation)']

    if technologies is None:
        raise ValueError(f"Unknown technology group: {args.technologies}. Choose from 'nuclear', 'hydro', 'geothermal', or 'thermal'.")
    
    test_periods = [(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31'))]

    # ---------- Merge + label prep ----------
    train_val_df, test_df, feature_names, target_columns, integer_encoding = ppd.preprocess_data(failure_data_path=args.failures,
                                                                                                weather_data_path=args.weather,
                                                                                                power_load_data_path=args.powerload,
                                                                                                feature_names=feature_names,
                                                                                                cyclic_features=["Season", "Month", "DayOfWeek", "DayOfYear"],
                                                                                                state_filter = args.states,
                                                                                                state_one_hot=True,
                                                                                                initial_MC_state_filter=args.initial_state,
                                                                                                final_MC_state_target = args.final_state,
                                                                                                technology_filter=technologies,
                                                                                                technology_one_hot=True,
                                                                                                test_periods=test_periods,
                                                                                                dropNA = True,
                                                                                                feature_na_drop_threshold = 0.2,
                                                                                                )

    if "Initial_gen_state" in feature_names:
        feature_names.remove("Initial_gen_state")
    # subset_length = 100
    # train_val_df = train_val_df.iloc[0:subset_length].copy().reset_index(drop=True)
    print(f"Train/Val Dataset shape: {train_val_df.shape}")


    
    # Standardize all continuous features (exclude one-hots and raw categorical/cyclic markers)
    exclude = {"Holiday", "Weekend", "Season", "Month", "DayOfWeek", "DayOfYear"}
    stand_cols = [f for f in feature_names if not f.startswith("State_") and not f.startswith("Technology_") and not f.endswith("_isnan") and not f.endswith("_sin") and not f.endswith("_cos") and f not in exclude]

    print(f"Standardized features ({len(stand_cols)}): {stand_cols}")

    feature_names.sort()
    stand_cols.sort()
    target_columns.sort()
    

    # ---------- Define model specs ----------
    # For both models, we define:
    # - common_build: parameters used in all builds
    # - build_grid:   parameters to search at build time
    # - common_train: parameters used in all training runs
    # - train_grid:   parameters to search at training time

    # 1) ######### XGBoost #########
    xgb_common_build = {
                        "feature_cols" : feature_names,
                        "target_cols"  : target_columns,
                        "eval_metric"  : "mlogloss",
                        "objective"    : "multi:softprob",
                        "num_classes"  : 3,
                        "early_stopping_rounds" : 10,
                        "device"          : args.device,     
                        }
    xgb_common_prepare_data = {"train_ratio": 0.80, 
                            "val_ratio": 0.2,
                            "standardize":stand_cols, 
                            "reweight_train_data_density":'Temperature'}
    xgb_common_train = {
                        "weights_data": True,
                     }
    
    xgb_build_grid = {
                        "max_depth":   [6, 8, 10], #[4, 6, 8, 10]
                        "eta":         [0.04, 0.06, 0.08], # [0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
                        "gamma":       [0.4, 0.6, 0.8], #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        "reg_lambda":  [0.4,  0.6,  0.8],
                        "subsample"       : [0.7, 0.8, 0.9],
                        "num_boost_round" : [100, 200] #[100, 200, 500, 800]
                        }
    xgb_prepare_data_grid = {
                                "reweight_power": [0.0, 0.5, 1.0, 1.5, 2.0]
                            }


    # 2) ######### MLP #########
    mlp_common_build = {
        "feature_cols": feature_names,
        "target_cols":  target_columns,
        "num_classes": 3 if args.final_state == "all" else 2,
    }

    mlp_common_prepare_data = {"train_ratio": 0.80, 
                               "val_ratio": 0.2,
                               "standardize":stand_cols, 
                               "reweight_train_data_density":'Temperature'}
    
    mlp_common_train =  {
                        "optimizer": "adam",
                        "loss": "focal_loss",
                        "regularization_type": "L2",
                        "weights_data": True,
                        "device": args.device,
                        "early_stopping": False,
                        "grad_clip_norm": 1.0,
                        }
    

    mlp_build_grid = {
        "hidden_sizes": [
                        # (256, 512, 256, 128, 64), # 5 layers
                        # (512, 512, 256, 128, 64), # 5 layers
                        # (128, 128, 128, 128, 64), # 5 layers
                        (256, 512, 512, 256, 128, 64, 64, 64, 64, 64, ) # 10 layers
                        # (128, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,) # 12 layers
                        ],
        "activations": [
                        # ("relu",) * 5,
                        ("relu",) * 10,
                        # ("relu",) * 12,
                        ],
    }
    
    mlp_prepare_data_grid = {
                                "reweight_power": [0.0, 0.5, 1.0, 1.5, 2.0]
                            }
    
    mlp_train_grid = {
                    "epochs"              : [100],   # upper bound — levels will cap
                    "batch_size"          : [512],
                    "lambda_reg"          : [5e-2, 1e-3, 5e-3], # 1e-3
                    "lr"                  : [5e-5, 1e-4, 2e-4, 4e-4, 1e-3, 2e-3],
                    "focal_loss_alpha"    : [[1.0, 1.0, 1.0]],#[0.01, 0.985, 0.985],[1.0, 1.0, 1.0]
                    "focal_loss_alpha_schedule" : ['constant'],  # ['constant', 'linear', 'exponential', 'cosine']
                    "focal_loss_gamma"    : [0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], # [0.0, 0.5, 1, 1.5, 2.0, 2.5, 3.0]
                    "focal_loss_gamma_schedule" : ['constant'], # ['constant', 'linear', 'exponential', 'cosine']
                    "lr_scheduler"          : ['exponential', 'cosine'],  # ['constant', 'linear', 'exponential', 'cosine']
                    # "patience"            : [20],
                    # "min_delta"           : [5e-5],
                    # "flat_delta"          : [1e-4],
                    # "flat_patience"       : [50],
                    # "flat_mode"           : ['iqr'],
                    # "rel_flat"            : [2e-3],
                    # "burn_in"             : [150],
                    }
    #                 Stop only when BOTH conditions hold:
    #                   • No-improve: best hasn't improved by min_delta for `patience` epochs, AND
    #                   • Flat-window: range in last `flat_patience` epochs <= flat_delta (abs or relative).


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
            "common_prepare_data": xgb_common_prepare_data,
            "prepare_data_grid":   xgb_prepare_data_grid,
            "common_train": xgb_common_train,
            "train_grid":   {},
        })
    if include_mlp:
        model_specs.append({
            "name":         "MLP",
            "constructor":  lambda: im.MLP(verbose=False),
            "common_build": mlp_common_build,
            "build_grid":   mlp_build_grid,
            "common_prepare_data": mlp_common_prepare_data,
            "prepare_data_grid":   mlp_prepare_data_grid,
            "common_train": mlp_common_train,
            "train_grid":   mlp_train_grid,
        })

    if not model_specs:
        raise SystemExit("No models selected. Use --models xgb|mlp|both")

    # Validation metric per model
    val_metric = {spec["name"]: "cross_entropy" for spec in model_specs}
    val_metric = {name:metric for name, metric in val_metric.items() if any(spec["name"]==name for spec in model_specs)}

    # Training levels (successive halving caps)
    n_rows = len(train_val_df)
    # training_levels = [
    #     {"name": "L1-fast",   "epochs": 200,  "data_cap": int(n_rows * 0.50)},
    #     {"name": "L2-medium", "epochs": 200,  "data_cap": int(n_rows * 0.75)},
    #     {"name": "L3-full",   "epochs": 400, "data_cap": None},
    # ]

    training_levels = [
        # {"name": "L1-fast",   "epochs": 15,  "data_cap": int(n_rows * 0.50)},
        {"name": "L3-full",   "epochs": 100, "data_cap": None},
    ]

    # ---------- Run search ----------
    winners = gs.successive_halving_search(
        model_specs=model_specs,
        data=train_val_df,
        # standardize=stand_cols,
        result_csv=str(args.result_csv)+f"_{''.join(args.models)}_{args.states}_{args.technologies}_{args.initial_state}_{args.final_state}"+".csv",
        # train_ratio=1.0 - args.val_frac,
        # val_ratio=args.val_frac,
        val_metric_per_model=val_metric,
        levels=training_levels,
        top_keep_ratio=args.top_keep,
        resume=args.reuse_results,
        # reweight_train_data_density='Temperature',
        seed=args.seed,
        # seed=args.seed,  # if supported by your helper
    )

    # ---------- Summarize results ----------
    # print("\n=== Global winner ===")

    # best_model = min(winners, key=lambda x: x[3])
    # mi, build_p, train_p, score = best_model
    # spec = model_specs[mi]
    # print(f"[{spec['name']}] score={score:.6f}")
    # print(" build:", json.dumps(build_p, sort_keys=True))
    # print(" train:", json.dumps(train_p, sort_keys=True))
    # print("-" * 60)


if __name__ == "__main__":
    main()