import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import os
import itertools


data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'DATA/raw')


def load_events(events_per_year = None, years = None, columns = None):
    """
    Load events from a CSV file and return a DataFrame.
    INPUTS:
    events_per_year: int, number of events to load per year
    years: list of str, years to load (e.g., ['2018', '2019'])
    columns: list of str, columns to load (e.g., ['column1', 'column2'])
    OUTPUTS:
    events_df: DataFrame, loaded events
    """
    if years is not None:
        years = [str(year) for year in years]
    filenames_events = []
    for filename in os.listdir(os.path.join(data_folder, 'GADS-NERC')):
        if not filename.startswith('GADS_') and not filename.endswith('.xlsx'):
            continue
        split_filename = filename.split('_')
        if len(split_filename)>1 and filename.split('_')[1] == 'Events':
            file_year = filename.split('_')[2]
            if (years is not None) and (file_year not in years):
                continue
            filenames_events.append(filename)

    filenames_events.sort()
    events_df = pd.DataFrame()
    for filename in filenames_events:
        try:
            year = filename.split('_')[2]
            print(f"Loading Events : {year}")
            filepath = os.path.join(data_folder, 'GADS-NERC', filename)
            cols = columns if columns is not None else pd.read_excel(filepath, nrows=1).columns.tolist()
            
            if events_per_year is not None:
                new_df = pd.read_excel(filepath, nrows=events_per_year, usecols=cols)
            else:
                new_df = pd.read_excel(filepath)
            # Find intersection of columns between existing events_df and new_df
            if not events_df.empty:
                common_cols = list(set(events_df.columns) & set(new_df.columns))
                events_df = pd.concat([events_df[common_cols], new_df[common_cols]], ignore_index=True)
                ignore_cols = list(set(new_df.columns) - set(common_cols))
                if ignore_cols:
                    print(f"Warning: Ignoring columns {ignore_cols} in {filename}")
            else:
                events_df = new_df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    print(f"Loaded {events_df.shape[0]} events")
    return events_df

def load_performances(perf_per_year = None, years = None, columns = None):
    """
    Load performance from a CSV file and return a DataFrame.
    INPUTS:
    perf_per_year: int, number of rows to load per year
    years: list of str, years to load (e.g., ['2018', '2019'])
    columns: list of str, columns to load (e.g., ['column1', 'column2'])
    OUTPUTS:
    perf_df: DataFrame, loaded events
    """
    filenames_perf = []
    for filename in os.listdir(os.path.join(data_folder, 'GADS-NERC')):
        if not filename.startswith('GADS_'):
            continue
        split_filename = filename.split('_')
        if len(split_filename)>1 and filename.split('_')[1] == 'Performance':
            if years is not None and filename.split('_')[2] not in years:
                continue
            filenames_perf.append(filename)

    filenames_perf.sort()
    perf_df = pd.DataFrame()
    for filename in filenames_perf:
        try:
            year = filename.split('_')[2]
            print(f"Loading Performances : {year}")
            filepath = os.path.join(data_folder, 'GADS-NERC', filename)
            cols = columns if columns is not None else pd.read_excel(filepath, nrows=1).columns.tolist()
            
            if perf_per_year is not None:
                new_df = pd.read_excel(filepath, nrows=perf_per_year, usecols=cols)
            else:
                new_df = pd.read_excel(filepath)
            # Find intersection of columns between existing events_df and new_df
            if not perf_df.empty:
                common_cols = list(set(perf_df.columns) & set(new_df.columns))
                perf_df = pd.concat([perf_df[common_cols], new_df[common_cols]], ignore_index=True)
                ignore_cols = list(set(new_df.columns) - set(common_cols))
                if ignore_cols:
                    print(f"Warning: Ignoring columns {ignore_cols} in {filename}")
            else:
                perf_df = new_df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    print(f"Loaded {perf_df.shape[0]} events")
    return perf_df


def filter_events(events_df, CauseCodes=['U1', 'U2', 'U3', 'D1', 'D2', 'D3'], filter_fuel=True, exclude_states=['Other','Mexico','South America'], include_states=None):
    """
    Clean the events DataFrame by
    1. Keeping only unplanned outages and deratings (U1, U2, U3, D1, D2, D3)
    2. Removing duplicate events
    3. If filter_fuel is True, keeping only events with gas as the fuel type
    4. Adding the unit's location (state and region)
    """
    # 1. Keep only desired events (unplanned outages)
    events_df = events_df.loc[events_df['EventTypeCode'].isin(CauseCodes)].copy()


    # 2. Remove duplicates
    units_df = pd.read_excel(os.path.join(data_folder, 'GADS-NERC', 'GADS_Unit_2024_20250505.xlsx'))
    events_df['EventStartDT'] = events_df['EventStartDT'].astype('datetime64[ns]')
    events_df['EventEndDT'] = events_df['EventEndDT'].astype('datetime64[ns]')

    events_filtered = {}

    for _, row in tqdm(events_df.iterrows(), total=events_df.shape[0], desc='Merging duplicated events'):
        eventID = row['EventID']
        if eventID not in events_filtered:
            events_filtered[row['EventID']] = row
        else:
            # If the event already exists, we merge the data
            events_filtered[eventID]['EventStartDT'] = min(events_filtered[eventID]['EventStartDT'], row['EventStartDT'])
            events_filtered[eventID]['EventEndDT'] = max(events_filtered[eventID]['EventEndDT'], row['EventEndDT'])

    events_filtered_df = pd.DataFrame.from_dict(events_filtered, orient='index').reset_index(drop=True)

    # 3. Keep events with the right fuel type
    units_performance = load_performances()

    fuel_used_failure = []
    for _, row in tqdm(events_filtered_df.iterrows(), total=events_filtered_df.shape[0], desc='Matching fuels'):
        unitid = row['UnitID']
        start = row['EventStartDT']
        perf = units_performance.loc[(units_performance['UnitID'] == unitid)&(units_performance['ReportingYearNbr']==start.year)&(units_performance['ReportingMonthNbr']==start.month)]
        if perf.empty:
            # print(f'No performance data for unit {unitid} at {start}')
            fuel_used_failure.append(None)
            continue
        fuel = perf['FuelCodeName1'].values[0] if perf['FuelSequenceName1'].values[0] == 'Primary Fuel' else perf['FuelCodeName2'].values[0] if perf['FuelSequenceName2'].values[0] == 'Primary Fuel' else None
        fuel_used_failure.append(fuel)

    events_filtered_df['FuelFailure'] = fuel_used_failure

    if filter_fuel:
        gas_fuels = ['Gas', 'Propane', 'Other - Gas (Cu. Ft.)', 'Other Gas (Cu Ft)']
        events_filtered_df = events_filtered_df.loc[events_filtered_df['FuelFailure'].isin(gas_fuels)].copy()
        events_filtered_df.sort_values(by='EventStartDT', inplace=True)

    # 4. get the unit's location
    States = []
    Region = []

    for _, row in tqdm(events_filtered_df.iterrows(), total=events_filtered_df.shape[0], desc='Adding state and region'):
        unitid = row['UnitID']
        unit = units_df.loc[units_df['UnitID'] == unitid]
        if unit.empty:
            States.append(None)
            Region.append(None)
            continue
        States.append(unit['StateName'].values[0])
        Region.append(unit['RegionCode'].values[0])
        
    events_filtered_df['State'] = States
    events_filtered_df['Region'] = Region

    # Exclude events from certain states
    if exclude_states is not None:
        events_filtered_df = events_filtered_df.loc[~events_filtered_df['State'].isin(exclude_states)].copy()
    
    # Include only events from certain states
    if include_states is not None:
        events_filtered_df = events_filtered_df.loc[events_filtered_df['State'].isin(include_states)].copy()

    return events_filtered_df

def num_units_gas():
    units_performance = load_performances()
    units_df = pd.read_excel('DATA/GADS-NERC/GADS_Unit_2024_20250505.xlsx')

    gas_fuels = ['Gas', 'Propane', 'Other - Gas (Cu. Ft.)', 'Other Gas (Cu Ft)']

    num_units_gas = []
    states = units_df['StateName'].unique().tolist()
    states.sort()
    years = units_performance['ReportingYearNbr'].unique().tolist()
    years.sort()
    months = units_performance['ReportingMonthNbr'].unique().tolist()
    months.sort()

    main_fuel = []
    for _, row in tqdm(units_performance.iterrows(), total=units_performance.shape[0], desc='Processing main fuel'):
        # Determine the main fuel based on the primary fuel sequence
        fuel = row['FuelCodeName1'] if row['FuelSequenceName1'] == 'Primary Fuel' else row['FuelCodeName2'] if row['FuelSequenceName2'] == 'Primary Fuel' else None
        main_fuel.append(fuel)
    units_performance['MainFuel'] = main_fuel
    units_performance = units_performance.loc[units_performance['MainFuel'].isin(gas_fuels)].copy()

    for year, month, state in tqdm(itertools.product(years, months, states), total=len(years)*len(months)*len(states), desc='Counting gas units'):
        perf_month_fuel = units_performance.loc[(units_performance['ReportingYearNbr'] == year) & (units_performance['ReportingMonthNbr'] == month) & (units_performance['UnitID'].isin(units_df.loc[units_df['StateName']==state, 'UnitID']))]

        num_units_gas.append([year, month, state, perf_month_fuel.shape[0]])

    num_units_gas_df = pd.DataFrame(num_units_gas, columns=['Year', 'Month', 'State', 'NumUnitsGas'])

    return num_units_gas_df

