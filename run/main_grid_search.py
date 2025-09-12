#!/usr/bin/env python
import os, json, argparse
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import inferenceModels as im    # your file



def parse_args():
    p = argparse.ArgumentParser(description="Successive-halving grid search for MLP & XGBoost.")
    p.add_argument("--failures",      type=str, default="../DATA/filtered_events.csv")
    p.add_argument("--events",        type=str, default="../DATA/event_count.csv")
    p.add_argument("--weather",       type=str, default="../DATA/weather_data_per_state_all.csv")
    p.add_argument("--powerload",     type=str, default="../DATA/power_load_input.csv")
    p.add_argument("--target",        type=str, default="Frequency", choices=["Unit_Failure","Frequency"])
    p.add_argument("--clusters",      type=int, default=1)
    p.add_argument("--result_csv",    type=str, default="../Results/grid_search_log.csv")
    p.add_argument("--top_keep",      type=float, default=0.33)
    p.add_argument("--val_frac",      type=float, default=0.2)
    p.add_argument("--no_resume",     action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.result_csv), exist_ok=True)

    # ---------- Load base feature set ----------
    weather = pd.read_csv(args.weather, index_col=[0,1], parse_dates=[0])
    power   = pd.read_csv(args.powerload, index_col=[0,1], parse_dates=[0])
    feature_names = list(weather.columns) + list(power.columns) + \
                    ['Season','Month','DayOfWeek','DayOfYear','Holiday','Weekend']
    feature_names = list(set(feature_names)-set(['EventStartDT', 'Date', 'PRCP_30dz']))

    feature_names.sort()
    print(f"Initial features ({len(feature_names)}): {feature_names}")

    # ---------- Merge + label prep ----------
    merged_df, feature_cols, target_cols = im.preprocess_data(
        failure_path=args.failures,
        event_count_path=args.events,
        weather_path=args.weather,
        power_load_path=args.powerload,
        feature_names=feature_names,
        target=args.target,
        state_one_hot=True,
        cyclic_features=['Season', 'Month', 'DayOfWeek', 'DayOfYear'],
        cause_code_n_clusters=args.clusters,
        feature_na_drop_threshold=0.10
    )
    # merged_df, feature_cols, target_cols = [],[],[]
    # stand_cols = []
    


    # Standardize all non-binary + non-state one-hot
    stand_cols = [f for f in feature_cols if not f.startswith("State_") and f not in ["Holiday","Weekend", 'Season','Month','DayOfWeek','DayOfYear']]
    print(f"Standardized features: {stand_cols}")
    

    # ---------- Define model specs ----------
    # 1) XGBoost
    xgb_common_build = {
        "feature_cols": feature_cols,
        "target_cols":  target_cols,
        "eval_metric":  "logloss",# if args.target == "Unit_Failure" else "rmse",
        "objective":    "reg:logistic",# if args.target == "Unit_Failure" else "reg:squarederror",
        "early_stopping_rounds": 10,
        "subsample": 1.0,
        "num_boost_round": 500,
        "device": "cuda",     # change to 'cuda' if your xgboost build supports it
    }
    xgb_build_grid = {
        "max_depth":   [4, 6, 8],
        "eta":         [0.02, 0.05, 0.1],
        "gamma":       [0.0, 0.25, 0.5, 0.75, 1.0],
        "lambda_reg":  [0.0, 0.25, 0.5, 0.75, 1.0],
        # num_boost_round can also be searched if you want:
        # "num_boost_round": [300, 500, 800],
        # "subsample": [0.7, 1.0],
    }
    # XGBoost "train" stage: only weights flag (others handled at build or by sklearn fit)
    xgb_train_grid = {
        "weights_data": [True],
    }

    # 2) MLP
    mlp_common_build = {
        "feature_cols": feature_cols,
        "target_cols":  target_cols,
        "out_act_fn":   "sigmoid",# if args.target == "Unit_Failure" else None,
    }
    mlp_build_grid = {
        "hidden_sizes": [
            # (128, 128, 64),
        #    (256, 128, 64),
        #    (256, 256, 128, 64),
           (256, 256, 128, 64),
        ],
        "activations": [
            # ("relu","relu","relu"),
        #    ("relu","relu","relu","relu"),
           ("relu","relu","relu","relu","relu"),
        ],
    }
    mlp_train_grid = {
        "optimizer": ["adam"],
        "loss":   ["logloss"],# if args.target == "Unit_Failure" else ["mse"],
        "regularization_type": ["L2"],
        "lambda_reg": [5e-5, 1e-4, 2e-4, 4e-4, 1e-3],
        "epochs": [1000],            # upper bound — levels will cap
        "batch_size": [128, 256],
        "lr": [1e-4, 2e-4, 4e-4, 1e-3],
        "device": ["cuda"],           # set to 'cuda' if available else 'cpu
        "weights_data": [True],
    }

    model_specs = [
        {
            "name": "xgboostModel",
            "constructor": lambda: im.xgboostModel(verbose=False),
            "common_build": xgb_common_build,
            "build_grid":   xgb_build_grid,
            "common_train": {},
            "train_grid":   xgb_train_grid,
        },
        # {
        #     "name": "MLP",
        #     "constructor": lambda: im.MLP(verbose=False),
        #     "common_build": mlp_common_build,
        #     "build_grid":   mlp_build_grid,
        #     "common_train": {},
        #     "train_grid":   mlp_train_grid,
        # }
    ]

    # Validation metric per model
    val_metric = {
        "xgboostModel": "logloss",# if args.target == "Unit_Failure" else "mse",
        "MLP":          "logloss" #if args.target == "Unit_Failure" else "mse"
    }

    winners = im.successive_halving_search(
        model_specs=model_specs,
        data=merged_df,
        standardize=stand_cols,
        result_csv=args.result_csv,
        train_ratio=1.0 - args.val_frac,
        val_ratio=args.val_frac,
        val_metric_per_model=val_metric,
        levels=[
            {"name":"L1-fast",   "epochs": 150,  "data_cap": int(len(merged_df)*0.4)},
            {"name":"L2-medium", "epochs": 500,  "data_cap": int(len(merged_df)*0.8)},
            {"name":"L3-full",   "epochs": 2000, "data_cap": None},
        ],
        top_keep_ratio=args.top_keep,
        resume=True#not args.no_resume
    )

    # print("\n=== Top survivors at final level ===")
    # for (mi, build_p, train_p, score) in winners:
    #     print(f"[{model_specs[mi]['name']}] score={score:.6f}")
    #     print(" build:", json.dumps(build_p, sort_keys=True))
    #     print(" train:", json.dumps(train_p,  sort_keys=True))
    #     print("-"*60)

    print("\n=== Global winner ===")

    best_model = min(winners, key=lambda x: x[3])
    mi, build_p, train_p, score = best_model
    print(f"[{model_specs[mi]['name']}] score={score:.6f}")
    print(" build:", json.dumps(build_p, sort_keys=True))
    print(" train:", json.dumps(train_p,  sort_keys=True))
    print("-"*60)


if __name__ == "__main__":
    main()
