from typing import Dict,List,Tuple

import numpy as np
import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar






def preprocess_data(
    failure_data_path: str,
    weather_data_path: str,
    power_load_data_path: str,
    feature_names: List[str],
    initial_MC_state_filter: str = 'all',
    technology_filter: List[str] = None,
    state_one_hot: bool = True,
    technology_one_hot: bool = True,
    state_filter: str = 'all',
    cyclic_features: List[str] = None,
    dropNA: bool = True,
    feature_na_drop_threshold: float = 0.2,
    test_periods: List[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], Dict[str, Dict[str, int]]]:
    """
    Preprocess and merge generator failure, weather, and power-load datasets
    at (Datetime_UTC, State) resolution, then optionally split out test periods.

    High-level steps:
      1. Load failure, weather and power-load CSVs.
      2. Normalize datetime/timezone and state formats.
      3. Apply filters: technology, initial Markov state, state_filter.
      4. Select requested features that actually exist; drop columns with too many NaNs.
         - Weather: keep 'Heat_index' / 'Wind_chill' but add *_isnan flags and fill NaNs.
      5. Merge failure + weather + load on (Datetime_UTC, State).
      6. Optionally drop rows with remaining NaNs.
      7. Group duplicate rows (multiple units) by discrete keys and sum Data_weight.
      8. Encode State and Technology (one-hot or integer codes).
      9. Add calendar features (Season, Month, DayOfWeek, DayOfYear, Holiday, Weekend).
     10. Encode Final_gen_state as integer target.
     11. Apply cyclic encoding to selected features.
     12. Cast features to float32, targets to int32, Data_weight to float32.

    Parameters
    ----------
    failure_data_path : str
        Path to the unit-level failure data CSV.
    weather_data_path : str
        Path to the per-state daily weather data CSV.
    power_load_data_path : str
        Path to the per-state daily power load data CSV.
    feature_names : List[str]
        Candidate feature columns from weather/power/load (plus optional State/Technology/etc.).
    initial_MC_state_filter : str
        If not 'all', keep only rows with this initial Markov state.
    technology_filter : List[str], optional
        If provided, keep only units whose Technology is in this list.
    state_one_hot : bool
        If True, one-hot encode State_* columns; otherwise use integer codes.
    technology_one_hot : bool
        If True, one-hot encode Technology_* columns; otherwise use integer codes.
    state_filter : str
        If not 'all', restrict to a single state (by two-letter code, case-insensitive),
        and do NOT use State as a feature.
    cyclic_features : List[str], optional
        Names of scalar features (e.g. 'Month', 'DayOfYear') to encode as sin/cos pairs.
    dropNA : bool
        If True, drop rows with any remaining NaNs after merging (on feature columns).
    feature_na_drop_threshold : float
        Drop any weather/load feature with a NaN fraction strictly greater than this.
    test_periods : List[Tuple[pd.Timestamp, pd.Timestamp]], optional
        List of (start_date, end_date) tuples (inclusive, in UTC or naive) used to
        carve out an explicit test set. If None, all data is returned in train_val_df.
    keep_initial_state : bool
        If True, keep and return Initial_gen_state as a column (integer-coded).

    Returns
    -------
    train_val_df : pd.DataFrame
        Processed data NOT in test periods, with columns:
        ['Datetime_UTC', (optional 'Initial_gen_state'), features..., target(s), 'Data_weight'].
    test_df : pd.DataFrame
        Processed data WITHIN test periods (same column structure as train_val_df).
    final_feature_names : List[str]
        List of feature columns after all encodings and filtering.
    target_columns : List[str]
        List of target column names (currently ['Final_gen_state']).
    integer_encoding : Dict[str, Dict[str, int]]
        Mapping from original string labels to integer codes for categorical columns
        (e.g. 'States', 'Technologies', 'Final_gen_state').
    """

    # ----------------- defensive copies & setup -----------------
    cyclic_features = list(cyclic_features or [])
    # Capitalize first letter of every feature name for consistency with loaded CSVs
    feature_names = [
        (name[0].upper() + name[1:]) if isinstance(name, str) and name else name
        for name in feature_names
    ]
    feature_names = list(set(feature_names))  # remove duplicates

    integer_encoding: Dict[str, Dict[str, int]] = {}

    # ----------------- 1) Load failure data -----------------
    failure_df = pd.read_csv(failure_data_path, parse_dates=["Datetime_UTC"])
    # Normalize column capitalization
    failure_df.columns = [
        (name[0].upper() + name[1:]) if isinstance(name, str) and name else name
        for name in failure_df.columns
    ]

    # Standardize column names
    if "Geographical State" in failure_df.columns:
        failure_df = failure_df.rename(columns={"Geographical State": "State"})
    if "Count" in failure_df.columns:
        failure_df = failure_df.rename(columns={"Count": "Data_weight"})
    if "Data_weight" not in failure_df.columns:
        failure_df["Data_weight"] = 1.0

    # Normalize state codes
    if "State" in failure_df.columns:
        failure_df["State"] = failure_df["State"].astype(str).str.upper()

    # Technology filter
    if technology_filter is not None:
        failure_df = failure_df[failure_df["Technology"].isin(technology_filter)].copy()

    # Initial Markov state filter
    if "Initial_gen_state" in failure_df.columns:
        if initial_MC_state_filter != "all":
            failure_df = failure_df.loc[
                failure_df["Initial_gen_state"] == initial_MC_state_filter
            ].copy()
        else:
            feature_names.append("Initial_gen_state")


    # Restrict to a single state if requested
    if state_filter != "all" and "State" in failure_df.columns:
        failure_df = failure_df.loc[
            failure_df["State"] == state_filter.upper()
        ].copy()
        feature_names = [f for f in feature_names if f != "State"]

    # ----------------- 2) Load weather data -----------------
    weather_df = pd.read_csv(
        weather_data_path,
        parse_dates=["datetime"],
    )
    weather_df.columns = [
        (name[0].upper() + name[1:]) if isinstance(name, str) and name else name
        for name in weather_df.columns
    ]

    # Normalize MultiIndex to (Datetime_UTC, State) in UTC
    weather_df["State"] = weather_df["State"].astype(str).str.upper()
    dt = weather_df["Datetime"]
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")
    weather_df["Datetime_UTC"] = dt
    weather_df.drop(columns=["Datetime"], inplace=True)
    weather_df = weather_df.set_index(["Datetime_UTC", "State"])

    # Keep only requested weather features that exist
    keep_weather_features = (set(feature_names) & set(weather_df.columns)) - {
        "Datetime",
        "State",
    }
    weather_df = weather_df[list(sorted(keep_weather_features))].copy()

    # Drop weather columns with too many NaNs, except Heat_index/Wind_chill
    na_frac = weather_df.isna().mean()
    drop_cols = list(
        set(na_frac[na_frac > feature_na_drop_threshold].index.tolist())
        - {"Heat_index", "Wind_chill"}
    )
    if drop_cols:
        print(
            f"Dropping weather columns with >{np.around(feature_na_drop_threshold * 100)}% NaN: {drop_cols}"
        )
        weather_df.drop(columns=drop_cols, inplace=True)
        feature_names = [f for f in feature_names if f not in drop_cols]

    # Special handling for Heat_index and Wind_chill: keep, with *_isnan flags
    for col in ["Heat_index", "Wind_chill"]:
        if col in weather_df.columns and col in feature_names:
            isnan_col = f"{col}_isnan"
            weather_df[isnan_col] = weather_df[col].isna().astype(np.float32)
            feature_names.append(isnan_col)
            weather_df[col] = weather_df[col].fillna(0.0)


    # ----------------- 3) Load power load data -----------------
    power_load_df = pd.read_csv(
        power_load_data_path,
        parse_dates=["UTC time"],
    )
    power_load_df.columns = [
        (name[0].upper() + name[1:]) if isinstance(name, str) and name else name
        for name in power_load_df.columns
    ]

    # Normalize MultiIndex to (Datetime_UTC, State) in UTC
    power_load_df["State"] = power_load_df["State"].astype(str).str.upper()
    dt = power_load_df["UTC time"]
    if dt.dt.tz is None:
        power_load_df["Datetime_UTC"] = dt.dt.tz_localize("UTC")
    else:
        power_load_df["Datetime_UTC"] = dt.dt.tz_convert("UTC")
    power_load_df.drop(columns=["UTC time"], inplace=True)
    power_load_df = power_load_df.set_index(["Datetime_UTC", "State"])

    # Keep only requested power features that exist
    keep_power_features = set(feature_names) & set(power_load_df.columns)
    power_load_df = power_load_df[list(sorted(keep_power_features))].copy()

    # Drop load columns with too many NaNs
    na_frac = power_load_df.isna().mean()
    drop_cols = na_frac[na_frac > feature_na_drop_threshold].index.tolist()
    if drop_cols:
        print(
            f"Dropping power load columns with >{np.around(feature_na_drop_threshold * 100)}% NaN: {drop_cols}"
        )
        power_load_df.drop(columns=drop_cols, inplace=True)
        feature_names = [f for f in feature_names if f not in drop_cols]

    # ----------------- 4) Merge all data -----------------
    # Join weather & load onto failure records by (Datetime_UTC, State)
    merged_data = failure_df.join(
        weather_df, how="left", on=["Datetime_UTC", "State"]
    )
    merged_data = merged_data.join(
        power_load_df, how="left", on=["Datetime_UTC", "State"]
    )

    # Optionally drop rows with NaNs (on any column)
    if dropNA:
        merged_data = merged_data.dropna(axis=0, how="any").copy()

    merged_data.reset_index(drop=True, inplace=True)


    # ----------------- 5) Keep only relevant columns & aggregate -----------------
    # We want to aggregate duplicate rows (multiple units) by discrete keys
    # and sum Data_weight.
    discrete_keys = ["Datetime_UTC"]
    if "State" in merged_data.columns and "State" in feature_names:
        discrete_keys.append("State")
    if initial_MC_state_filter == "all" and "Initial_gen_state" in merged_data.columns:
        discrete_keys.append("Initial_gen_state")
    if "Final_gen_state" in merged_data.columns:
        discrete_keys.append("Final_gen_state")
    if "Technology" in merged_data.columns and "Technology" in feature_names:
        discrete_keys.append("Technology")

    # Keys we want to keep explicitly before encoding
    feat_keep = ["Datetime_UTC", "Final_gen_state"]
    if "State" in merged_data.columns and "State" in feature_names:
        feat_keep.append("State")
    if "Technology" in merged_data.columns and "Technology" in feature_names:
        feat_keep.append("Technology")

    # Add all candidate feature names
    feat_keep = [f for f in (feat_keep + feature_names) if f in merged_data.columns]
    feat_keep = list(set(feat_keep))  # remove duplicates
    cols_for_group = list(dict.fromkeys(feat_keep + ["Data_weight"]))  # preserve order

    merged_data = merged_data[cols_for_group].copy()

    # Aggregate by discrete keys; assume non-key columns identical per key
    agg_dict = {col: "first" for col in merged_data.columns if col not in ["Data_weight"] + discrete_keys}
    agg_dict["Data_weight"] = "sum"

    merged_data = (
        merged_data.groupby(discrete_keys, as_index=False)
        .agg(agg_dict)
        .reset_index(drop=True)
    )


    # ----------------- 6) State encoding -----------------
    if "State" in merged_data.columns:
        if state_filter != "all":
            # Single state -> do not use as feature
            if "State" in feature_names:
                feature_names.remove("State")
        else:
            if state_one_hot:
                merged_data = pd.get_dummies(
                    merged_data,
                    columns=["State"],
                    drop_first=False,
                    dtype=int,
                )
                if "State" in feature_names:
                    feature_names.remove("State")
                feature_names += [c for c in merged_data.columns if c.startswith("State_")]
            else:
                if "State" not in feature_names:
                    feature_names.append("State")
                if isinstance(merged_data.loc[merged_data.index[0], "State"], str):
                    cats = {
                        s: i
                        for i, s in enumerate(
                            np.sort(merged_data["State"].astype(str).unique())
                        )
                    }
                    merged_data["State"] = merged_data["State"].map(cats)
                    integer_encoding["States"] = cats

    # ----------------- 7) Technology encoding -----------------
    if "Technology" in merged_data.columns:
        if technology_one_hot:
            merged_data = pd.get_dummies(
                merged_data,
                columns=["Technology"],
                drop_first=False,
                dtype=int,
            )
            if "Technology" in feature_names:
                feature_names.remove("Technology")
            feature_names += [
                c for c in merged_data.columns if c.startswith("Technology_")
            ]
        else:
            if "Technology" not in feature_names:
                feature_names.append("Technology")
            if isinstance(merged_data.loc[merged_data.index[0], "Technology"], str):
                cats = {
                    s: i
                    for i, s in enumerate(
                        np.sort(merged_data["Technology"].astype(str).unique())
                    )
                }
                merged_data["Technology"] = merged_data["Technology"].map(cats)
                integer_encoding["Technologies"] = cats

    # ----------------- 8) Calendar features -----------------
    # Season
    if "Season" in feature_names:
        def get_season(ts: pd.Timestamp) -> float:
            ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
            Y = ts.year
            seasons = {
                0.0: (pd.Timestamp(f"{Y}-03-20", tz="UTC"), pd.Timestamp(f"{Y}-06-20", tz="UTC")),  # Spring
                1.0: (pd.Timestamp(f"{Y}-06-21", tz="UTC"), pd.Timestamp(f"{Y}-09-22", tz="UTC")),  # Summer
                2.0: (pd.Timestamp(f"{Y}-09-23", tz="UTC"), pd.Timestamp(f"{Y}-12-20", tz="UTC")),  # Autumn
                3.0: (pd.Timestamp(f"{Y}-12-21", tz="UTC"), pd.Timestamp(f"{Y+1}-03-19", tz="UTC")),  # Winter
            }
            for s, (start, end) in seasons.items():
                if start <= ts <= end:
                    return s
            return 3.0

        merged_data["Season"] = merged_data["Datetime_UTC"].apply(get_season)

    if "Month" in feature_names:
        merged_data["Month"] = merged_data["Datetime_UTC"].dt.month

    if "DayOfWeek" in feature_names:
        merged_data["DayOfWeek"] = merged_data["Datetime_UTC"].dt.dayofweek

    if "DayOfYear" in feature_names:
        merged_data["DayOfYear"] = merged_data["Datetime_UTC"].dt.dayofyear

    if "Holiday" in feature_names:
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(
            start=merged_data["Datetime_UTC"].min(),
            end=merged_data["Datetime_UTC"].max(),
        )
        merged_data["Holiday"] = merged_data["Datetime_UTC"].dt.normalize().isin(holidays)

    if "Weekend" in feature_names:
        merged_data["Weekend"] = merged_data["Datetime_UTC"].dt.weekday >= 5

    # ----------------- 9) Target construction -----------------
    target_columns = ["Final_gen_state"]
    if isinstance(merged_data.loc[merged_data.index[0], "Final_gen_state"], str):
        cats = {
            s: i
            for i, s in enumerate(
                np.sort(merged_data["Final_gen_state"].astype(str).unique())
            )
        }
        merged_data["Final_gen_state"] = merged_data["Final_gen_state"].map(cats)
        integer_encoding["Final_gen_state"] = cats
    if "Initial_gen_state" in feature_names:
        gen_state_encoding = integer_encoding.get("Final_gen_state", {})
        merged_data["Initial_gen_state"] = merged_data["Initial_gen_state"].map(gen_state_encoding)

    # ----------------- 10) Cyclic feature encoding -----------------
    for feat in list(cyclic_features):
        if feat not in merged_data.columns:
            # Skip quietly if requested cyclic feature was not created
            continue
        series = merged_data[feat]
        min_val, max_val = series.min(), series.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            merged_data[f"{feat}_sin"] = 0.0
            merged_data[f"{feat}_cos"] = 1.0
        else:
            phase = 2.0 * np.pi * (series - min_val) / (max_val - min_val)
            merged_data[f"{feat}_sin"] = np.sin(phase)
            merged_data[f"{feat}_cos"] = np.cos(phase)
        if feat in feature_names:
            feature_names.remove(feat)
        feature_names += [f"{feat}_sin", f"{feat}_cos"]
        merged_data.drop(columns=[feat], inplace=True, errors="ignore")

    # ----------------- 11) Final column selection & dtypes -----------------
    # Preserve original feature order where possible

    feature_names = [f for f in feature_names if f in merged_data.columns]

    cols = feature_names + target_columns + ["Data_weight"]
    if initial_MC_state_filter == "all" and "Initial_gen_state" in merged_data.columns:
        cols = ["Initial_gen_state"] + cols
    cols = ["Datetime_UTC"] + cols

    cols = list(set(cols))  # remove duplicates

    for col in cols:
        if col not in merged_data.columns:
            raise KeyError(f"Expected column '{col}' not found in merged data.")

    # Cast types
    for tcol in target_columns:
        merged_data[tcol] = merged_data[tcol].astype(np.int32)

    for fcol in feature_names:
        merged_data[fcol] = merged_data[fcol].astype(np.float32)

    merged_data["Data_weight"] = merged_data["Data_weight"].astype(np.float32)

    # Sort by time
    merged_df = merged_data.sort_values(by="Datetime_UTC").reset_index(drop=True)

    # ----------------- 12) Train/test split by periods -----------------
    if test_periods:
        mask_test = pd.Series(False, index=merged_df.index)
        for start_date, end_date in test_periods:
            sd = pd.to_datetime(start_date)
            ed = pd.to_datetime(end_date)

            # inclusive end-of-day
            ed = ed.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

            if sd.tzinfo is None:
                sd = sd.tz_localize("UTC")
            else:
                sd = sd.tz_convert("UTC")
            if ed.tzinfo is None:
                ed = ed.tz_localize("UTC")
            else:
                ed = ed.tz_convert("UTC")

            mask = (merged_df["Datetime_UTC"] >= sd) & (merged_df["Datetime_UTC"] <= ed)
            mask_test |= mask

        test_df = merged_df.loc[mask_test].copy().reset_index(drop=True)
        train_val_df = merged_df.loc[~mask_test].copy().reset_index(drop=True)
    else:
        # no explicit test periods -> empty test set
        test_df = merged_df.iloc[0:0].copy().reset_index(drop=True)
        train_val_df = merged_df.copy().reset_index(drop=True)


    # Final column ordering for both splits
    train_val_df = train_val_df[cols].copy()
    test_df = test_df[cols].copy()


    return train_val_df, test_df, feature_names, target_columns, integer_encoding