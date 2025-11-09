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
    """ Main function to train the ML model. """
    t_start = time()

    args = parse_args()         # Arguments
    np.random.seed(args.seed)   # Reproducibility

    # I/O prep
    ensure_inputs_exist(args.failures, args.weather, args.powerload)
    print(f"Computing transition probabilities starting with {args.models} from state {args.initial_state} for {args.technologies} generators.")

    # ---------- Get feature set ----------
    feature_names = load_feature_bases(args.weather, args.powerload)
    # feature_names = ['Temperature']
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
    technologies = ['Gas Turbine/Jet Engine (Simple Cycle Operation)']
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
    

    if "xgb" in args.models:

        build_kw = {"device": "cuda", "early_stopping_rounds": 10, "eta": 0.08, "eval_metric": "mlogloss", "feature_cols": ["State_KANSAS", "Technology_Fluidized Bed", "State_IDAHO", "State_WISCONSIN", "Technology_Co-generator Block ", "State_ARIZONA", "State_CONNECTICUT", "24h_max_load", "Technology_CoG steam units ", "Season_sin", "State_ILLINOIS", "Technology_CC steam units", "Season_cos", "Tmin", "Technology_Fossil-Steam", "FDD", "State_MONTANA", "State_MINNESOTA", "State_TENNESSEE", "2d_load_sum", "State_TEXAS", "Extreme_cold", "State_NEW JERSEY", "State_NORTH CAROLINA", "State_MISSOURI", "Precip_3d", "Sea_level_pressure", "CDD", "Tmean", "Technology_Internal Combustion/Reciprocating Engines", "State_OREGON", "State_NEW MEXICO", "State_SOUTH CAROLINA", "Relative_humidity", "State_DELAWARE", "Temperature", "State_ALABAMA", "Heat_index_isnan", "HDD", "State_CALIFORNIA", "State_RHODE ISLAND", "1d_load_sum", "State_KENTUCKY", "FDD3d", "Weekend", "Extreme_heat", "Tmax", "State_MARYLAND", "Hourly_load_change", "State_NEBRASKA", "Precip_1d", "Dew_point_temperature", "State_LOUISIANA", "DayOfYear_sin", "State_MASSACHUSETTS", "State_UTAH", "State_SOUTH DAKOTA", "State_INDIANA", "State_MISSISSIPPI", "State_FLORIDA", "Technology_CoG GT units", "Technology_Gas Turbine/Jet Engine (Simple Cycle Operation)", "State_MAINE", "Load", "State_OKLAHOMA", "DayOfWeek_cos", "Wet_bulb_temperature", "DayOfYear_cos", "State_WEST VIRGINIA", "State_ARKANSAS", "Heat_index", "State_OHIO", "State_WYOMING", "Month_cos", "State_NEW HAMPSHIRE", "24h_min_load", "Month_sin", "State_GEORGIA", "Technology_CC GT units ", "State_VERMONT", "Snow_depth", "State_NEW YORK", "State_NORTH DAKOTA", "State_IOWA", "Technology_Combined Cycle Block", "Precipitation", "State_WASHINGTON", "Wind_chill_isnan", "Pressure_3hr_change", "Wind_speed", "Technology_Multi-boiler/Multi-turbine", "Technology_Gas Turbine/Jet Engine with HSRG", "State_PENNSYLVANIA", "CDD3d", "State_MICHIGAN", "State_VIRGINIA", "State_NEVADA", "Holiday", "Wind_chill", "HDD3d", "DayOfWeek_sin", "Station_level_pressure", "State_COLORADO"], "gamma": 0.8, "max_depth": 6, "num_boost_round": 100, "num_classes": 3, "objective": "multi:softprob", "reg_lambda": 0.6, "subsample": 0.9, "target_cols": ["Final_gen_state"]}
        train_kw = {"weights_data": true}

        xgb_model = im.xgboostModel(verbose=False)
        # xgb_model.build_model(max_depth=6,
        #                     eta=0.07,
        #                     gamma=0.8,
        #                     reg_lambda=1,
        #                     num_boost_round=100,
        #                     feature_cols=feature_names,
        #                     target_cols=target_columns,
        #                     num_classes=3,
        #                     eval_metric='mlogloss', # rmse, logloss, mae, mape
        #                     objective='multi:softprob',
        #                     subsample=1,
        #                     device=args.device,
        #                     early_stopping_rounds=10)
        xgb_model.build_model(**build_kw)
        

        xgb_model.prepare_data(train_val_df, train_ratio=0.80, val_ratio=0.2, standardize=stand_cols, reweight_train_data_density='Temperature')

        xgb_model.train_model(weights_data=True)

        path_to_save = THIS_DIR / "../Results/Models" / f"XGB_model_{args.technologies}_{args.initial_state}.pth"

        xgb_model.save_model(path_to_save)


    elif "mlp" in args.models:

        build_kw = {"activations": ["relu", "relu", "relu", "relu", "relu"], "feature_cols": ["1d_load_sum", "24h_max_load", "24h_min_load", "2d_load_sum", "CDD", "CDD3d", "DayOfWeek_cos", "DayOfWeek_sin", "DayOfYear_cos", "DayOfYear_sin", "Dew_point_temperature", "Extreme_cold", "Extreme_heat", "FDD", "FDD3d", "HDD", "HDD3d", "Heat_index", "Heat_index_isnan", "Holiday", "Hourly_load_change", "Load", "Month_cos", "Month_sin", "Precip_1d", "Precip_3d", "Precipitation", "Pressure_3hr_change", "Relative_humidity", "Sea_level_pressure", "Season_cos", "Season_sin", "Snow_depth", "State_ALABAMA", "State_ARIZONA", "State_ARKANSAS", "State_CALIFORNIA", "State_COLORADO", "State_CONNECTICUT", "State_DELAWARE", "State_FLORIDA", "State_GEORGIA", "State_IDAHO", "State_ILLINOIS", "State_INDIANA", "State_IOWA", "State_KANSAS", "State_KENTUCKY", "State_LOUISIANA", "State_MAINE", "State_MARYLAND", "State_MASSACHUSETTS", "State_MICHIGAN", "State_MINNESOTA", "State_MISSISSIPPI", "State_MISSOURI", "State_MONTANA", "State_NEBRASKA", "State_NEVADA", "State_NEW HAMPSHIRE", "State_NEW JERSEY", "State_NEW MEXICO", "State_NEW YORK", "State_NORTH CAROLINA", "State_NORTH DAKOTA", "State_OHIO", "State_OKLAHOMA", "State_OREGON", "State_PENNSYLVANIA", "State_SOUTH CAROLINA", "State_SOUTH DAKOTA", "State_TENNESSEE", "State_TEXAS", "State_UTAH", "State_VERMONT", "State_VIRGINIA", "State_WASHINGTON", "State_WEST VIRGINIA", "State_WISCONSIN", "State_WYOMING", "Station_level_pressure", "Temperature", "Tmax", "Tmean", "Tmin", "Weekend", "Wet_bulb_temperature", "Wind_chill", "Wind_chill_isnan", "Wind_speed"], "hidden_sizes": [256, 512, 256, 128, 64], "num_classes": 3, "target_cols": ["Final_gen_state"]}
        train_kw = {"batch_size": 512, "burn_in": 150, "device": "cuda", "early_stopping": True, "epochs": 30, "flat_delta": 0.0001, "flat_mode": "iqr", "flat_patience": 50, "grad_clip_norm": 1.0, "lambda_reg": 0.001, "loss": "cross_entropy", "lr": 0.0004, "lr_scheduler": "plateau", "min_delta": 5e-05, "optimizer": "adam", "patience": 20, "regularization_type": "L2", "rel_flat": 0.002, "weights_data": True}


        mlp_model = im.MLP(verbose=False)
        # mlp_model.build_model(feature_cols=feature_names,
                            # target_cols=target_columns,
                            # num_classes=3,
                            # hidden_sizes=(128, 128, 64, 32, 32, 32, 32, 32),
                            # activations=("relu",) * 8)
        mlp_model.build_model(**build_kw)

        mlp_model.prepare_data(train_val_df, train_ratio=0.80, val_ratio=0.2, standardize=stand_cols, reweight_train_data_density='Temperature')
        
        mlp_model.train_model(**train_kw)
        # mlp_model.train_model(optimizer='adam', 
        #                       loss='cross_entropy',
        #                         regularization_type='L2', lambda_reg=1e-3,
        #                         weights_data=True,
        #                         epochs=100, batch_size=512, lr=2e-3,
        #                         device=args.device,
        #                         # smart stopping knobs
        #                         early_stopping=True, patience=20, min_delta=5e-5,
        #                         flat_delta=1e-4, flat_patience=50, flat_mode='iqr', rel_flat=2e-3, burn_in=150,
        #                         # stability & LR policy
        #                         grad_clip_norm=1.0,
        #                         lr_scheduler='plateau', scheduler_kwargs={'factor':0.5, 'patience':3, 'min_lr':1e-6})


        path_to_save = THIS_DIR / "../Results/Models" / f"MLP_model_{args.technologies}_{args.initial_state}.pth"
        mlp_model.save_model(path_to_save)
        print(mlp_model.val_loss)

    t_end = time()
    print(f"Training completed in {t_end - t_start:.1f} seconds.")




if __name__ == "__main__":
    main()