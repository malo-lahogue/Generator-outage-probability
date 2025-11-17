import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import itertools


def clean_state_name(state_col: str) -> str:
    s = state_col.split('_')[1]
    clean_s = ''
    for w in s.split( ' '):
        clean_s += w.capitalize() + ' '
    state = clean_s.strip()
    return state

def aggregate_scenario(gen_states_df: pd.DataFrame) -> pd.DataFrame:
    """Agregates individaul generators states into a regional unavailable generation capacity scenario.
    Assumes 
    - Each generator has a capacity of 1 unit.
    - Derates level is random uniform between 0 and 1.
    - Derate level is constant during each outage event.
    INPUTS:
    - gen_states_df: DataFrame with columns ['Datetime_UTC', 'Gen_state', 'UnitID']
    OUTPUTS:
    - scenario_df: DataFrame with columns ['Datetime_UTC', 'Unavailable_capacity']
    """
    # Validate input
    assert 'Datetime_UTC' in gen_states_df.columns, "gen_states_df must contain 'Datetime_UTC' column"
    assert 'Gen_state' in gen_states_df.columns, "gen_states_df must contain 'Gen_state' column"
    assert 'UnitID' in gen_states_df.columns, "gen_states_df must contain 'UnitID' column"


    scenario_df = pd.DataFrame()
    scenario_df['Datetime_UTC'] = gen_states_df['Datetime_UTC'].unique()
    scenario_df = scenario_df.sort_values('Datetime_UTC').reset_index(drop=True)
    scenario_df['Unavailable_capacity'] = 0.0
    scenario_df['Num_units'] = 0

    for unitID, unit_df in gen_states_df.groupby('UnitID'):
        if len(unit_df) != unit_df['Datetime_UTC'].nunique():
            raise ValueError(f"Duplicate Datetime_UTC entries found for UnitID {unitID}")
        
        # For each unit, determine derate levels during outages
        unit_df = unit_df.sort_values('Datetime_UTC').reset_index(drop=True)
        unit_df['Derate_level'] = 0.0  # Default derate level is 0 (fully available)

        outage_mask = unit_df['Gen_state'] == 'U'
        unit_df.loc[outage_mask, 'Derate_level'] = 1.0

        derate_mask = unit_df['Gen_state'] == 'D'
        derate_mask_ = np.concatenate([np.array([0]), derate_mask.to_numpy(dtype=int)])
        derate_change_mask = derate_mask_[1:]-derate_mask_[:-1]
        derate_start_idx = np.where(derate_change_mask == 1)[0]
        derate_end_idx = np.where(derate_change_mask == -1)[0]
        derate_start_end_idx = list(zip(derate_start_idx, derate_end_idx))
        
        for derate_start, derate_end in derate_start_end_idx:
            event_mask = (unit_df.index >= derate_start) & (unit_df.index < derate_end)
            derate_level = np.random.uniform(0, 1)
            unit_df.loc[event_mask, 'Derate_level'] = derate_level
        
        # Add unit's unavailable capacity to scenario
        # scenario_df =   scenario_df.merge(unit_df[['Datetime_UTC', 'Derate_level']],
        #                                  on='Datetime_UTC',
        #                                  how='left',
        #                                  suffixes=('', f'_{unitID}'))
        corresponding_dates_mask = scenario_df['Datetime_UTC'].isin(unit_df['Datetime_UTC'])
        scenario_df.loc[corresponding_dates_mask, 'Unavailable_capacity'] += unit_df['Derate_level'].values
        scenario_df.loc[corresponding_dates_mask, 'Num_units'] += 1
    
    scenario_df['Unavailable_capacity (%)'] = (scenario_df['Unavailable_capacity'] / scenario_df['Num_units']) * 100.0

        
    return scenario_df

def get_scenario_inputs(feature_inputs_df, test_failures_df):
    scenarios_inputs_by_state = {}

    for s in tqdm([c for c in feature_inputs_df.columns if c.startswith('State_')], desc="Getting feature inputs by state"):
        state = clean_state_name(s)
        scenarios_inputs_by_state[state] = feature_inputs_df.loc[feature_inputs_df[s] == 1].drop_duplicates().sort_values('Datetime_UTC').copy()

    generators_per_state = {}
    for state in tqdm(test_failures_df['State'].unique(), desc="Getting generators by state"):
        state_failures_df = test_failures_df[test_failures_df['State'] == state].copy()
        state_gens = {'UnitID': [], 'Technology': [], 'Start_date': [], 'End_date': []}
        for unitID, unit_df in state_failures_df.groupby('UnitID'):
            start = unit_df['Datetime_UTC'].min()
            end = unit_df['Datetime_UTC'].max()
            state_gens['UnitID'].append(unitID)
            state_gens['Technology'].append(unit_df['Technology'].iloc[0])
            state_gens['Start_date'].append(start)
            state_gens['End_date'].append(end)

        generators_per_state[state] = pd.DataFrame(state_gens)

    return scenarios_inputs_by_state, generators_per_state

def get_stationary_distribution(transition_probability_matrix: np.ndarray) -> np.ndarray:
    """Computes the stationary distribution of a Markov chain given its transition probability matrix.
    INPUTS:
    - transition_probability_matrix: 2D numpy array of shape (n_states, n_states)
    OUTPUTS:
    - stationary_distribution: 1D numpy array of shape (n_states,)
    """
    assert transition_probability_matrix.ndim == 2, "Transition probability matrix must be 2D"
    assert transition_probability_matrix.shape[0] == transition_probability_matrix.shape[1], "Transition probability matrix must be square"
    
    n_states = transition_probability_matrix.shape[0]
    A = np.transpose(transition_probability_matrix) - np.eye(n_states)
    A = np.vstack([A, np.ones(n_states)])
    b = np.zeros(n_states + 1)
    b[-1] = 1.0

    A = A.astype(float)
    b = b.astype(float)

    stationary_distribution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return stationary_distribution

def generate_unavailable_capacity_scenario_per_gen(
    covariates_inputs_df: pd.DataFrame,
    generators_data_df: pd.DataFrame,
    models: dict,
    num_scenarios: int = 1000,
    min_scenarios_per_gen: int = 10,
) -> dict:
    """
    Generate per-generator state scenarios from ML-based transition models.

    Internals:
    - Uses integer encoding for states: 0 -> 'A', 1 -> 'D', 2 -> 'U'.
    - Vectorizes:
        * Transition probabilities into a (3, T, 3) array.
        * Scenario simulation over K scenarios per generator.

    INPUTS:
    - covariates_inputs_df : DataFrame with columns
        ['Datetime_UTC'] + features needed by each model.
      Must be sorted by Datetime_UTC (if not, we sort internally).
    - generators_data_df : DataFrame with at least columns
        ['UnitID', 'Start date', 'End date'].
    - models : dict mapping initial-state label -> fitted model, e.g.
        {'A': model_A, 'D': model_D, 'U': model_U}
      Each model must expose `.feature_cols` and `.predict(X)` returning
      probs with columns ordered as [P(A), P(D), P(U)].
    - num_scenarios : Target number of *aggregate* scenarios. We generate
      K scenarios per generator such that K^G ≳ num_scenarios, where
      G = number of generators.

    OUTPUT:
    - scenarios_per_gen : dict[unit_id -> list[pd.DataFrame]]
        Each DataFrame has columns ['Datetime_UTC', 'Gen_state', 'UnitID'].
    """
    # ---- State encoding ----
    state_labels = ["A", "D", "U"]
    label_to_idx = {s: i for i, s in enumerate(state_labels)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    idx_to_label[-1] = "N/A"  # for uninitialized states
    n_states = len(state_labels)

    # ---- Ensure time ordering + extract times as numpy ----
    cov_df = covariates_inputs_df.sort_values("Datetime_UTC").reset_index(drop=True)
    times = cov_df["Datetime_UTC"].to_numpy()
    T = len(times)

    # ---- Precompute transition probabilities as numpy array ----
    # P_all[from_state, t, to_state] with shape (3, T, 3)
    P_all = np.empty((n_states, T, n_states), dtype=np.float64)

    for from_label, model in models.items():
        if from_label not in label_to_idx:
            raise KeyError(f"Unexpected initial state key in models: {from_label}")
        from_idx = label_to_idx[from_label]

        X_inputs = cov_df[model.feature_cols]
        probs = np.asarray(model.predict(X_inputs), dtype=np.float64)  # (T, 3)

        if probs.ndim != 2 or probs.shape[1] != n_states:
            raise ValueError(
                f"Model for state {from_label} must return (T, {n_states}) "
                f"probability array; got shape {probs.shape}."
            )

        # Normalize defensively to avoid any numerical drift
        row_sums = probs.sum(axis=1, keepdims=True)
        # If any row sum is zero, fall back to uniform
        probs = np.divide(
            probs,
            row_sums,
            out=np.full_like(probs, 1.0 / n_states),
            where=row_sums > 0,
        )
        P_all[from_idx, :, :] = probs

    # ---- How many scenarios per generator? ----
    # num_scenarios_per_gen ** num_generators >= num_scenarios 
    num_generators = max(1, len(generators_data_df))
    num_scenarios_per_gen = int(np.ceil(num_scenarios ** (1.0 / num_generators)))
    num_scenarios_per_gen = max(num_scenarios_per_gen, min_scenarios_per_gen)

    rng = np.random.default_rng()

    scenarios_per_gen: dict = {}

    # ---- Loop over generators (outer loop only, time & scenarios are vectorized) ----
    for unit in tqdm(
        generators_data_df.itertuples(index=False),
        total=len(generators_data_df),
        desc="Generating per unit scenarios"):
        
        # Expect these attribute names to match DataFrame column names
        unit_id = getattr(unit, "UnitID")
        start_date = getattr(unit, "Start_date")
        end_date = getattr(unit, "End_date")

        # Mask time window for this unit
        mask = (times >= start_date) & (times <= end_date)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            # No coverage for this unit in covariate time grid
            scenarios_per_gen[unit_id] = []
            continue

        idx_start = idx[0]
        idx_end = idx[-1] + 1  # slice end (exclusive)
        # times_unit = times[idx_start:idx_end]
        # T_u = len(times_unit)

        # Slice transition probs for this unit: shape (3, T_u, 3)
        # P_unit = P_all[:, idx_start:idx_end, :]

        # ---- Initial stationary distribution at t=0 from P_unit[:,0,:] ----
        trans0 = P_all[:, idx_start, :]  # (3, 3)
        pi0 = get_stationary_distribution(trans0)  # (3,)

        # Clip + renormalize for numerical safety
        pi0 = np.clip(pi0, 0.0, None)
        s = pi0.sum()
        if s <= 0:
            pi0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            pi0 = pi0 / s

        K = num_scenarios_per_gen
        # simulate only inside [idx_start, idx_end)
        states = np.full((K, T), -1, dtype=np.int8)

        # initial state at idx_start
        states[:, idx_start] = rng.choice(n_states, size=K, p=pi0)

        # ---- Inhomogeneous Markov chain simulation (vectorized over K) ----
        for t in range(idx_start+1, idx_end):
            last_states = states[:, t - 1]  # shape (K,)

            # Gather transition rows for each scenario:
            # P_unit[from_state, t, :] -> (K, 3)
            probs_t = P_all[last_states, t, :]  # (K, 3)

            # Renormalize each row just in case
            row_sums = probs_t.sum(axis=1, keepdims=True)
            probs_t = np.divide(
                probs_t,
                row_sums,
                out=np.full_like(probs_t, 1.0 / n_states),
                where=row_sums > 0,
            )

            # Sample next states with inverse-CDF trick
            u = rng.random(size=K)
            cdf = probs_t.cumsum(axis=1)  # (K, 3)
            # new_state = first j s.t. u <= cdf[j]; equivalent to count of (u > cdf)
            new_states = (u[:, None] > cdf).sum(axis=1).astype(np.int8)
            states[:, t] = new_states
        # ---- Convert integer states back to 'A'/'D'/'U' per scenario ----
        unit_scenarios = []
        for k in range(K):
            gen_state_labels = [idx_to_label[int(s)] for s in states[k]]
            df_k = pd.DataFrame(
                {
                    "Datetime_UTC": times,
                    "Gen_state": gen_state_labels,
                    "UnitID": unit_id,
                }
            )
            unit_scenarios.append(df_k)
        scenarios_per_gen[unit_id] = unit_scenarios
        
    return scenarios_per_gen

def generate_unavailable_capacity_scenario(
    scenarios_per_gen: dict,
    num_scenarios: int = 1,
    seed: int = 42
):
    """
    Combine per-generator scenarios into aggregated unavailable capacity scenarios.
    Vectorized version.

    INPUT:
        scenarios_per_gen: dict[unitID -> list[pd.DataFrame]]
        num_scenarios: how many aggregated scenarios to produce
    OUTPUT:
        list of pd.DataFrame, each with:
            ['Datetime_UTC', 'Unavailable_capacity', 'Unavailable_capacity (%)']
    """
    rng = np.random.default_rng(seed)

    # ---- 1) Convert per-gen scenarios into uniform integer-coded arrays ----
    unit_ids = list(scenarios_per_gen.keys())
    G = len(unit_ids)

    # Determine the global time axis T
    # (all generators must share exact same Datetime_UTC grid)
    all_timesteps = set()
    for scen_list_gen in scenarios_per_gen.values():
        # Assume all scenarios for a given gen share the same time axis
        all_timesteps.update(scen_list_gen[0]['Datetime_UTC'].to_numpy())

    times = np.sort(np.array(list(all_timesteps)))
    T = len(times)

    # Build (G, K, T) integer matrix of states
    # Where each entry is 0, 1, or 2
    state_label_to_int = {"A": 0, "D": 1, "U": 2, "N/A": -1}

    K_per_gen = [len(scenarios_per_gen[u]) for u in unit_ids]
    if len(set(K_per_gen)) != 1:
        raise ValueError("All generators must have same number of per-gen scenarios.")

    K = K_per_gen[0]

    states = np.empty((G, K, T), dtype=np.int8)

    for gi, unit in enumerate(unit_ids):
        dfs = scenarios_per_gen[unit]
        for k, df in enumerate(dfs):
            if len(df) != df['Datetime_UTC'].nunique():
                raise ValueError(f"Duplicate Datetime_UTC entries found for UnitID {unit}")
            states[gi, k, :] = df["Gen_state"].map(state_label_to_int).to_numpy()

    # ---- 2) Randomly select which gen-scenario each generator uses in each agg scenario ----
    # choices: shape (num_scenarios, G), each ∈ {0...K-1}
    choices = rng.integers(0, K, size=(num_scenarios, G))

    # ---- 3) Build aggregated state tensor ----
    # Combine by selecting states[gen_idx, choice[scenario,gen], :]
    # Result: shape (num_scenarios, G, T)
    agg_states = np.empty((num_scenarios, G, T), dtype=np.int8)
    for si in range(num_scenarios):
        for gi in range(G):
            agg_states[si, gi, :] = states[gi, choices[si, gi], :]

    # ---- 4) Convert aggregated states to unavailable capacity ----
    #  state=2 (U): contributes 1.0
    #  state=1 (D): contributes α (sampled per derate block)
    #  state=0 (A): contributes 0
    #
    # First mark U contribution:
    U_mask = (agg_states == 2).astype(np.float32)

    # Next handle derates
    D_mask = (agg_states == 1)

    # Existing generator
    E_mask = (agg_states >= 0)

    # For each derate "block" (contiguous sequence), sample α ∈ [0.15,1]
    # block detection: diff along time axis
    scenarios_list = []
    for si in tqdm(range(num_scenarios),
                   total=num_scenarios,
                   desc="Generating unavailable capacity scenarios"):
        # α vector for (G,T)
        alpha_vals = np.zeros((G, T), dtype=np.float32)

        for gi in range(G):
            d = D_mask[si, gi, :]
            # detect transitions
            dd = np.diff(d.astype(np.int8), prepend=0)
            starts = np.where(dd == 1)[0]
            ends   = np.where(dd == -1)[0]
            if d[-1] == 1:
                ends = np.append(ends, T)

            for s, e in zip(starts, ends):
                a = rng.uniform(0.15, 1.0)
                alpha_vals[gi, s:e] = a

        # unavailable capacity per generator
        gen_unavail = U_mask[si] + alpha_vals

        # sum over generators
        u = gen_unavail.sum(axis=0)
        pct = (u / E_mask[si].sum(axis=0)) * 100

        df = pd.DataFrame({
            "Datetime_UTC": times,
            "Unavailable_capacity": u,
            "Unavailable_capacity (%)": pct
        })
        scenarios_list.append(df)

    return scenarios_list