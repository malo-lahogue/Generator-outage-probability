import os
import csv
import json
import math
import itertools
import datetime
from typing import Any, Dict, Iterable, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd


# ----------------- helpers -----------------


def _expand_grid(param_grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Turn a dict of lists into a sequence of dicts (Cartesian product).
    Example: {"a":[1,2], "b":[10]} -> {"a":1,"b":10}, {"a":2,"b":10}

    If param_grid is empty, yields a single empty dict.
    """
    keys = list(param_grid.keys())
    if not keys:
        yield {}
        return

    if set(keys) == {"hidden_sizes", "activations"}:
        hidden_sizes_list = param_grid["hidden_sizes"]
        activations_list = param_grid["activations"]

        if len(hidden_sizes_list) != len(activations_list):
            raise ValueError("When using 'hidden_sizes' and 'activations', both lists must have the same length.")

        for hs, act in zip(hidden_sizes_list, activations_list):
            yield {"hidden_sizes": hs, "activations": act}
    else:
        values_list: List[List[Any]] = []
        for k in keys:
            v = param_grid[k]
            if isinstance(v, (list, tuple, np.ndarray)):
                values_list.append(list(v))
            else:
                values_list.append([v])

        for combo in itertools.product(*values_list):
            yield dict(zip(keys, combo))


def _row_key(
    level_name: str,
    model_name: str,
    build_params: Dict[str, Any],
    data_params: Dict[str, Any],
    train_params: Dict[str, Any],
    state: Optional[str] = None,
) -> Tuple[str, str, str, str, Optional[str]]:
    """
    Deterministic key (including state) for resume logic.
    """
    build_str = json.dumps(build_params, sort_keys=True).replace(",", ";")
    data_str = json.dumps(data_params, sort_keys=True).replace(",", ";")
    train_str = json.dumps(train_params, sort_keys=True).replace(",", ";")
    return (str(level_name), str(model_name), build_str, data_str, train_str, state)


def _already_done_df(csv_path: str) -> pd.DataFrame:
    """
    Load previous results if any. Ensures required columns exist.
    """
    # base_cols = ["level", "model_name", "build_params", "data_params", "train_params", "min_val_loss", "timestamp"]
    # state_col = "state"

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
            # for c in base_cols + [state_col]:
            #     if c not in df.columns:
            #         df[c] = None
            # return df[base_cols + [state_col]]
        except Exception:
            pass
    return pd.DataFrame()
    # return pd.DataFrame(columns=base_cols + [state_col])


def _val_loss_numpy(
    model,
    metric: str = "cross_entropy",
    weight_col: str = "Data_weight",
    use_weights: bool = True,
) -> float:
    """
    Compute a (possibly weighted) validation loss for any model that has:
      - model.val_data  (pd.DataFrame or torch Dataset/Subsets)
      - model.feature_cols, model.target_cols
      - For DataFrame val_data: model.predict(X_df) -> np.ndarray (probs or preds)
      - For torch Dataset val_data: model implements `_gather_val_arrays()` and `_predict_np()`.

    Supported metrics:
        - 'mse', 'mae'       : generic regression-style losses (2D arrays).
        - 'logloss'          : binary BCE with probabilities.
        - 'cross_entropy'    : multi-class CE with probabilities.
    """

    metric_l = metric.lower()

    def _weighted_mean(per_sample: np.ndarray, w: Optional[np.ndarray]) -> float:
        per_sample = np.asarray(per_sample, dtype=np.float64).reshape(-1)
        if w is None:
            return float(per_sample.mean())
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        denom = w.sum()
        if denom <= 0:
            return float(per_sample.mean())
        return float((per_sample * w).sum() / denom)

    def _score_arrays(y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray]) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # ------- regression-style losses -------
        if metric_l in ("mse", "mae"):
            if y_true.ndim == 1:
                y_true = y_true[:, None]
            if y_pred.ndim == 1:
                y_pred = y_pred[:, None]
            if metric_l == "mse":
                per_sample = ((y_true - y_pred) ** 2).mean(axis=1)
                return _weighted_mean(per_sample, w)
            else:
                per_sample = np.abs(y_true - y_pred).mean(axis=1)
                return _weighted_mean(per_sample, w)

        # ------- binary logloss (BCE) -------
        if metric_l == "logloss":
            # y_true expected in {0,1}
            yt = y_true.astype(np.float64).reshape(-1)
            # y_pred is probability of class 1
            if y_pred.ndim == 2 and y_pred.shape[1] > 1:
                # assume last column is class-1 prob or use column 1
                p = y_pred[:, 1]
            else:
                p = y_pred.reshape(-1)

            eps = 1e-10
            p = np.clip(p, eps, 1.0 - eps)
            bce = -(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p))
            return _weighted_mean(bce, w)

        # ------- multi-class cross entropy -------
        if metric_l == "cross_entropy":
            # y_pred: (N, C) probabilities
            if y_pred.ndim == 1:
                # degenerate, treat as binary p(1)
                p = np.stack([1.0 - y_pred, y_pred], axis=1)
            else:
                p = y_pred

            if p.ndim != 2:
                raise ValueError("cross_entropy expects y_pred with shape (N, C).")

            # y_true can be class indices or one-hot
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                y_idx = y_true.argmax(axis=1)
            else:
                y_idx = y_true.reshape(-1).astype(int)

            n, c = p.shape
            if y_idx.shape[0] != n:
                raise ValueError(f"Length mismatch in CE: y_true={y_idx.shape[0]}, y_pred={n}")
            if (y_idx < 0).any() or (y_idx >= c).any():
                raise ValueError("y_true class indices out of range for y_pred.shape[1].")

            eps = 1e-10
            p = np.clip(p, eps, 1.0 - eps)
            p_true = p[np.arange(n), y_idx]
            ce = -np.log(p_true)
            return _weighted_mean(ce, w)

        raise ValueError("metric must be 'mse', 'mae', 'logloss' or 'cross_entropy'.")

    # ---- Path 1: pandas DataFrame val_data (e.g., xgboostModel) ----
    if hasattr(model, "val_data") and isinstance(model.val_data, pd.DataFrame):
        val_df: pd.DataFrame = model.val_data

        # columns
        X_cols = list(dict.fromkeys(np.array(model.feature_cols).flatten().tolist()))
        y_cols = list(dict.fromkeys(np.array(model.target_cols).flatten().tolist()))

        X_val = val_df[X_cols].copy()
        y_true = val_df[y_cols].to_numpy()

        # model.predict expected to return probabilities for classification
        y_pred = model.predict(X_val)

        # weights
        w = None
        if use_weights and weight_col in val_df.columns:
            w = val_df[weight_col].to_numpy(dtype=np.float64)

        return _score_arrays(y_true, y_pred, w)

    # ---- Path 2: torch Dataset val_data (e.g., MLP) ----
    has_helpers = all(
        hasattr(model, attr) for attr in ("_gather_val_arrays", "_predict_np")
    )
    if hasattr(model, "val_data") and not isinstance(model.val_data, pd.DataFrame) and has_helpers:
        # Use model's own helpers to keep y formatting consistent with loss
        X_val, y_val, W = model._gather_val_arrays(loss_name=metric_l)
        y_pred = model._predict_np(X_val)
        w = W if use_weights else None

        if hasattr(model, "_loss_np"):
            return model._loss_np(y_val, y_pred, weights=w, loss=metric_l)
        # Fallback to generic scorer
        return _score_arrays(y_val, y_pred, w)

    # ---- Fallback: use min validation loss history if present ----
    if hasattr(model, "val_loss") and len(model.val_loss) > 0:
        return float(np.min(model.val_loss))

    raise ValueError("Cannot compute validation loss: no suitable val_data or helpers found.")


# ----------------- core search over a single dataset (optionally for one state) -----------------


def _successive_halving_single(
    model_specs: List[Dict[str, Any]],
    data: pd.DataFrame,
    result_csv: str,
    val_metric_per_model: Optional[Dict[str, str]],
    levels: List[Dict[str, Any]],
    top_keep_ratio: float,
    num_folds_cv: int,
    resume: bool,
    subset_strategy: str,
    subset_seed: int,
    verbose: bool,
    seed: int,
    state_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Core successive halving over candidate hyperparameters for a single dataset.
    If state_name is not None, it's included in logging and returned in results.
    """

    # --- Build all (model_index, build_params, train_params) ---
    all_candidates: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for i, spec in enumerate(model_specs):
        build_grid = spec.get("build_grid", {}) or {}
        data_grid = spec.get("prepare_data_grid", {}) or {}
        train_grid = spec.get("train_grid", {}) or {}

        build_list = list(_expand_grid(build_grid))
        data_list = list(_expand_grid(data_grid))
        train_list = list(_expand_grid(train_grid))

        for b in build_list:
            for d in data_list:
                for t in train_list:
                    all_candidates.append((i, b, d, t))

    print(f"[grid] total candidates: {len(all_candidates)}")

    # --- Resume logic: build key -> score map ---
    done_df = _already_done_df(result_csv)
    # done_index: Dict[Tuple[str, str, str, str, Optional[str]], float] = {}
    done_index: Dict[Tuple, float] = {}

    if resume and len(done_df):
        for _, r in done_df.iterrows():
            state = str(r.get("state", None))
            if state == 'nan':
                state = None
            key = (
                str(r.get("level", None)),
                str(r.get("model_name", None)),
                str(r.get("build_params", None)),
                str(r.get("data_params", None)),
                str(r.get("train_params", None)),
                state
            )
            # try:
            s = r.get('score')
            s = s.split('[')[1].split(']')[0]
            if ',' in s:
                op = ', '
            else:
                op = '; '
            s = s.split(op)
            s = np.array(s, dtype=float).mean()
            done_index[key] = s
            # except Exception:
            #     continue
    survivors = list(all_candidates)

    # --- Iterate over levels ---
    for li, level in enumerate(levels):
        level_name = level["name"]
        level_epochs = int(level["epochs"])
        data_cap = level.get("data_cap", None)

        # subset data for this level
        if data_cap is not None and data_cap < len(data):
            if subset_strategy == "random":
                sub_data = data.sample(n=data_cap, random_state=subset_seed).reset_index(drop=True).copy()
            else:
                sub_data = data.iloc[:data_cap].reset_index(drop=True).copy()
        else:
            sub_data = data.reset_index(drop=True).copy()

        splits_idxs: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
        if num_folds_cv > 1:
            N = len(sub_data)
            indices = np.arange(N)
            np.random.seed(seed)
            np.random.shuffle(indices)
            fold_sizes = (N // num_folds_cv) * np.ones(num_folds_cv, dtype=int)
            fold_sizes[: N % num_folds_cv] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                val_idx = indices[start:stop]
                train_idx = np.concatenate([indices[:start], indices[stop:]])
                splits_idxs.append((train_idx, val_idx))
                current = stop
        else:
            print("No CV folds, taking 80/20 split")
            N = len(sub_data)
            indices = np.arange(N)
            np.random.seed(seed)
            np.random.shuffle(indices)
            split_at = int(N * 0.8)
            train_idx, val_idx = indices[:split_at], indices[split_at:]
            splits_idxs.append((train_idx, val_idx))

        scored: List[Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], float, str]] = []

        for (mi, build_params, data_params, train_params) in survivors:
            spec = model_specs[mi]
            mname = spec["name"]

            # Merge fixed + grid-specific params
            build_kw = dict(spec.get("common_build", {}))
            build_kw.update(build_params)

            data_kw = dict(spec.get("common_prepare_data", {}))
            data_kw.update(data_params)

            train_kw = dict(spec.get("common_train", {}))
            train_kw.update(train_params)

            # Cap epochs for MLP-like models that use 'epochs'
            if "epochs" in train_kw:
                train_kw["epochs"] = int(min(int(train_kw["epochs"]), level_epochs))

            # Resume key
            data_kw_key = data_kw.copy()
            data_kw_key.update({"k_folds": num_folds_cv})
            key_tuple = _row_key(level_name, mname, build_kw, data_kw_key, train_kw, state_name)
            build_str = key_tuple[2]
            data_str = key_tuple[3]
            train_str = key_tuple[4]


            if resume and key_tuple in done_index:
                score = done_index[key_tuple]
                scored.append((mi, build_params, data_params, train_params, score, mname))
                if verbose:
                    print(f"[resume] skip level={level_name} model={mname} state={state_name} -> score={score:.6g}")
                continue
            

            # ---------- Build / prepare / train ----------
            best_val_losses = []
            best_epochs = []
            val_losses_per_logits = defaultdict(list)
            for fold_i, split in enumerate(splits_idxs):

                model_obj = spec["constructor"]()
                model_obj.build_model(**build_kw)
                model_obj.prepare_data(data=sub_data, split_idxs=split, **data_kw)
                # ensure fresh weights if implemented
                if hasattr(model_obj, "reset_model"):
                    model_obj.reset_model()
                model_obj.val_loss = []

                model_obj.train_model(**train_kw)

                # ---------- Score ----------
                # Determine metric: default 'cross_entropy'
                if hasattr(model_obj, "val_loss"):
                    score = float(np.min(model_obj.val_loss))
                else:
                    metric_name = "cross_entropy"
                    if val_metric_per_model:
                        metric_name = (
                            val_metric_per_model.get(mname)
                            or val_metric_per_model.get(mname.lower())
                            or val_metric_per_model.get(model_obj.__class__.__name__)
                            or val_metric_per_model.get(model_obj.__class__.__name__.lower())
                            or metric_name
                        )

                    score = _val_loss_numpy(model_obj, metric=metric_name)

                best_val_losses.append(score)
                best_epoch = int(np.argmin(model_obj.val_loss)) + 1 if hasattr(model_obj, "val_loss") else -1
                best_epochs.append(best_epoch)
                if hasattr(model_obj, "val_loss_per_logit"):
                    for logit, vloss in model_obj.val_loss_per_logit.items():
                        val_losses_per_logits[logit].append(vloss)

            scored.append((mi, build_params, data_params, train_params, np.mean(best_val_losses), mname))

            score_dict = {
                "val_loss": best_val_losses,
                "best_epoch": best_epochs,
            }
            score_dict = json.dumps(score_dict, sort_keys=True).replace(",", ";")
            # ---------- Log row ----------
            if result_csv:
                os.makedirs(os.path.dirname(result_csv), exist_ok=True)
                row = {
                    "level": level_name,
                    "model_name": mname,
                    "build_params": build_str,
                    "data_params": data_str,
                    "train_params": train_str,
                    "state": state_name,
                    "val_loss_per_logit": val_losses_per_logits,
                    "score": score_dict,
                    "timestamp": datetime.datetime.now().astimezone().isoformat(),
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
        survivors = [(mi, bp, dp, tp) for (mi, bp, dp, tp, _, _) in scored[:keep]]

        # Final level: return winners in structured form
        if li == len(levels) - 1:
            winners: List[Dict[str, Any]] = []
            for (mi, bp, dp, tp, sc, mname) in scored[:keep]:
                winners.append(
                    {
                        "model_index": mi,
                        "model_name": mname,
                        "build_params": bp,
                        "data_params": dp,
                        "train_params": tp,
                        "score": sc,
                        "level": level_name,
                        "state": state_name,
                    }
                )
            return winners

    # Should never get here
    return []


# ----------------- main function -----------------


def successive_halving_search(
    model_specs: List[Dict[str, Any]],
    data: pd.DataFrame,
    result_csv: str,
    val_metric_per_model: Optional[Dict[str, str]] = None,
    levels: Optional[List[Dict[str, Any]]] = None,
    top_keep_ratio: float = 0.33,
    num_folds_cv: int =1,
    resume: bool = True,
    subset_strategy: str = "head",   # "head" or "random"
    subset_seed: int = 42,
    verbose: bool = False,
    model_per_state: bool = False,
    seed: int = 42,  # currently only used for data splitting (prepare_data)
) -> List[Dict[str, Any]]:
    """
    Successive halving over model families (MLP/XGBoost) and their grids.

    Returns a list of structured dicts:
        {
            "model_index": int,
            "model_name": str,
            "build_params": dict,
            "train_params": dict,
            "score": float,
            "level": str,
            "state": Optional[str],
        }

    Notes:
    - This implementation is **classification-oriented** and uses
      cross-entropy by default for validation scoring.
    - No warm-start: every candidate at every level is trained from scratch.
    - If model_per_state=True, the search is run separately for each state
      indicated by 'State_<STATE>' one-hot columns in `data`.
    """
    # --- default levels if none provided ---
    if levels is None:
        N = len(data)
        levels = [
            {"name": "L1-fast",   "epochs": 150,  "data_cap": int(N * 0.4)},
            {"name": "L2-medium", "epochs": 500,  "data_cap": int(N * 0.8)},
            {"name": "L3-full",   "epochs": 2000, "data_cap": None},
        ]

    # --- Per-state search wrapper ---
    if model_per_state:
        # Find all State_* columns
        states_list = [c.split("State_")[1] for c in data.columns if c.startswith("State_")]
        if verbose:
            print("Will search model per state")
            print(f"States found: {states_list}")

        if len(states_list) == 0:
            raise ValueError("model_per_state=True but no 'State_' columns found in data.")

        # Remove State_* from feature_cols in common_build (state is constant within each per-state dataset)
        for m in model_specs:
            feat_cols = m.get("common_build", {}).get("feature_cols", [])
            if feat_cols:
                m["common_build"]["feature_cols"] = [c for c in feat_cols if not str(c).startswith("State_")]

        all_results: List[Dict[str, Any]] = []

        # Use a consistent naming scheme: base + _state_<STATE>.csv
        base, ext = os.path.splitext(result_csv)
        if ext == "":
            ext = ".csv"

        for state in states_list:
            state_col = f"State_{state}"
            if state_col not in data.columns:
                continue

            # Restrict to rows for this state; keep all columns (State_* won't be used as features)
            data_state = data.loc[data[state_col] == 1].copy()
            if verbose:
                print(f"Searching models for state: {state} (N={len(data_state)})")

            state_csv = f"{base}_state_{state}{ext}"

            winners_state = _successive_halving_single(
                model_specs=model_specs,
                data=data_state,
                result_csv=state_csv,
                val_metric_per_model=val_metric_per_model,
                levels=levels,
                top_keep_ratio=top_keep_ratio,
                num_folds_cv=num_folds_cv,
                resume=resume,
                subset_strategy=subset_strategy,
                subset_seed=subset_seed,
                verbose=verbose,
                seed=seed,
                state_name=state,
            )
            all_results.extend(winners_state)

        return all_results

    # --- Single dataset (no per-state splitting) ---
    winners = _successive_halving_single(
        model_specs=model_specs,
        data=data,
        result_csv=result_csv,
        val_metric_per_model=val_metric_per_model,
        levels=levels,
        top_keep_ratio=top_keep_ratio,
        num_folds_cv=num_folds_cv,
        resume=resume,
        subset_strategy=subset_strategy,
        subset_seed=subset_seed,
        verbose=verbose,
        seed=seed,
        state_name=None,
    )

    return winners