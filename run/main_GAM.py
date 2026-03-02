# Import libraries

# Data processing and manipulation
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# from typing import Iterable, Any, Tuple, Dict
from pathlib import Path


# Custom models
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../src')))

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str((THIS_DIR / "../src").resolve()))

import preprocess_data as ppd
import GAMinferenceModels_V2 as gam_models



### Data

weather_data_file =  THIS_DIR/"../DATA/hourly/hourly_weather_by_state.csv"
power_load_file = THIS_DIR/"../DATA/hourly/hourly_load_by_state.csv"
failure_data_file = THIS_DIR/"../DATA/hourly/by_technology/hourly_failure_deltaTime_dataset_Gas_Turbine_Jet_Engine_(Simple_Cycle_Operation).csv"

print(failure_data_file)

all_data_df, _, feature_names, target_columns, integer_encoding = ppd.preprocess_data(failure_data_path=failure_data_file,
                                                                                        weather_data_path=weather_data_file,
                                                                                        power_load_data_path=power_load_file,
                                                                                        feature_names=['Temperature', 'Relative_humidity', 'Load', 'hours_in_state', 'State'],
                                                                                        cyclic_features=["Season", "Month", "DayOfWeek", "DayOfYear"],
                                                                                        state_one_hot=False,
                                                                                        initial_MC_state_filter='all',
                                                                                        # technology_filter=['Gas Turbine/Jet Engine (Simple Cycle Operation)'],
                                                                                        test_periods=None
                                                                                        )



# temporal features for regional classifiers
all_data_df['month_sin'] = np.sin(2*np.pi*all_data_df['Datetime_UTC'].dt.month/12)
all_data_df['month_cos'] = np.cos(2*np.pi*all_data_df['Datetime_UTC'].dt.month/12)

# Get list of states from one-hot encoded columns
idx2state = {v: k for k, v in integer_encoding['States'].items()}
all_data_df['State'] = all_data_df['State'].apply(lambda x: idx2state[x])
states_list = all_data_df['State'].unique().tolist()
states_list.sort()

print("States considered:", states_list)
print("Feature columns:", feature_names)
print("Length of dataset:", len(all_data_df))



##### 2022-2023 test

general_test_period = [(pd.Timestamp('2022-01-01 00:00:00', tz='UTC'), pd.Timestamp('2023-12-31 23:00:00', tz='UTC'))]
specific_test_periods_per_state = {state: general_test_period for state in states_list}
folder_name = "2022_2023_test_periods_highReg"

##### Train random split

spline_version = '5sALL_'
model_name = '_2MC_'
gamma = 0.5
# feature_cols = ['Temperature', 'Load_CDF', 'psi1', 'psi2', 'psi3', 'psi4', 'Hours_in_state']
# feature_cols = ['Temperature', 'Load_CDF', 'psi1', 'psi2', 'psi3', 'Hours_in_state']
# feature_cols = ['psi1', 'psi2', 'psi3']
feature_cols = ['Temperature', 'Load_CDF', 'Relative_humidity']
# feature_cols = ['Hours_in_state']

# zscore_cols=["psi1", "psi2", "psi3"]
zscore_cols = ['Temperature', 'Load_CDF', 'Relative_humidity']

# zscore_cols = []






transition_models, train_datasets, test_datasets, ess_res, scalers = gam_models.train_all_region_transition_models(all_data_df= all_data_df,
                                                                            regions= states_list[:1],
                                                                            feature_cols= feature_cols,
                                                                            zscore_cols=zscore_cols,
                                                                            regional_classifier_features= ['Temperature', 'Relative_humidity', 'Load_CDF', 'Temperature_3Dsum_hot', 'Temperature_3Dsum_cold', 'month_sin', 'month_cos'],
                                                                            # base_model_factory= base_model_factory,
                                                                            type_model='GAM',
                                                                            test_frac= 0.2,
                                                                            specific_test_periods= specific_test_periods_per_state,
                                                                            seed= 42,
                                                                            w_region_consider= False,
                                                                            w_stress_consider= False,
                                                                            gamma= gamma,
                                                                            clipping_quantile= 0.95,
                                                                            verbose= False)

print("Transition model successfully trained ")
print("Effective sample size results:", ess_res[states_list[0]], ess_res[states_list[0]]/len(all_data_df))


gam_models.export_transition_model_bundle(
    THIS_DIR/f"../Results/GAM/{folder_name}/{model_name}{spline_version}/model_bundle_gam.joblib",
    transition_models_by_region=transition_models,
    scalers_by_region=scalers,
    feature_cols=feature_cols,
    zscore_cols=zscore_cols,
    model_type="GAM",
    extra_metadata={"gamma": gamma, "clipping_quantile": 0.95},
)

print("Transition model bundle successfully exported")


gam_models.export_gam_predictions(
    transition_models= transition_models,
    test_datasets= test_datasets,
    scalers_by_region= scalers,
    cols_export=['Datetime_UTC', 'State', 'Stress', 'Initial_gen_state', 'Final_gen_state', 'Data_weight', 'pAD', 'pAO', 'pDA', 'pOA']+feature_cols,
    region_only=False,
    test=True,
    feature_cols= feature_cols,
    out_dir=THIS_DIR/f'../Results/GAM/{folder_name}/{model_name}{spline_version}/',
    model_name=model_name
)

print("Transition model successfully exported")

