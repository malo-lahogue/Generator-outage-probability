# Import libraries

# Data processing and manipulation
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from time import time

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
# rounded hours_in_state
# failure_data_file = THIS_DIR/"../DATA/hourly/by_technology/hourly_failure_deltaTime_dataset_Gas_Turbine_Jet_Engine_(Simple_Cycle_Operation).csv"
# Not rounded
failure_data_file = THIS_DIR/"../DATA/hourly/by_technology/hourly_failure_deltaTime_dataset_fullH_Gas_Turbine_Jet_Engine_(Simple_Cycle_Operation).csv"



print(failure_data_file)
t_start = time()

all_data_df, _, feature_names, target_columns, integer_encoding = ppd.preprocess_data(failure_data_path=failure_data_file,
                                                                                        weather_data_path=weather_data_file,
                                                                                        power_load_data_path=power_load_file,
                                                                                        # feature_names=['Temperature', 'Relative_humidity', 'precipitation', 'snow_depth', 'Load', 'State'],
                                                                                        feature_names=['Temperature', 'Relative_humidity', 'precipitation', 'snow_depth', 'Load', 'hours_in_state', 'State'],
                                                                                        # feature_names=['Temperature', 'Relative_humidity', 'Load', 'hours_in_state', 'State'],
                                                                                        # feature_names=['Temperature', 'Relative_humidity', 'Load',  'State'],
                                                                                        cyclic_features=["Season", "Month", "DayOfWeek", "DayOfYear"],
                                                                                        state_one_hot=False,
                                                                                        initial_MC_state_filter='all',
                                                                                        test_periods=None
                                                                                        )

# print(all_data_df.head())
# for col in all_data_df.columns:
#     print(col)
#     try:
#         print(f" has na : {all_data_df[col].isna().any()}")
#     except Exception as e:
#         print(e)
#     print(col)
#     try:
#         if pd.api.types.is_numeric_dtype(all_data_df[col]):
#             print(f" is finite : {np.isfinite(all_data_df[col].to_numpy()).all()}")
#     except Exception as e:
#         print(e)

t_end = time()
print(f"Data preprocessing completed in {t_end - t_start:.2f} seconds. Started at {pd.Timestamp(t_start, unit='s', tz='UTC')}, ended at {pd.Timestamp(t_end, unit='s', tz='UTC')}")
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



# ##### 2022-2023 test

general_test_period = [(pd.Timestamp('2022-01-01 00:00:00', tz='UTC'), pd.Timestamp('2023-12-31 23:00:00', tz='UTC'))]
specific_test_periods_per_state = {state: general_test_period for state in states_list}
folder_name = "2022_2023_test_periods_highReg"

# ##### Train random split

spline_version = '5sALL_'
# model_name = '_SMC_fullH_GAM_' #10402972
# model_name = '_MC_fullH_GAM_' #10402927
# model_name = '_AS_fullH_GAM_' #10403149
model_name = '_LogisticReg_fullH_GAM_' #10425454
folder = THIS_DIR/f"../Results/GAM/{folder_name}/{model_name}{spline_version}"
if not folder.exists():
    folder.mkdir(parents=True, exist_ok=True)
gamma = 0.5
# feature_cols = ['Temperature', 'Load_CDF', 'psi1', 'psi2', 'psi3', 'psi4', 'Hours_in_state']
# feature_cols = ['Temperature', 'Load_CDF', 'psi1', 'psi2', 'psi3', 'Hours_in_state']
# feature_cols = ['Temperature', 'Load_CDF', 'psi1', 'psi2', 'psi3', 'psi4']
# feature_cols = ['psi1', 'psi2', 'psi3', 'psi4', 'psi5', 'Hours_in_state']
# feature_cols = ['Temperature', 'Load_CDF', 'Relative_humidity']
# feature_cols = ['Temperature', 'Load_CDF', 'psi6', 'psi7']
feature_cols = ['Temperature', 'Precipitation', 'Snow_depth', 'Load_CDF', 'psi6', 'psi7', 'Hours_in_state']
# feature_cols = ['Temperature', 'Precipitation', 'Snow_depth', 'Load_CDF', 'psi6', 'psi7']
# feature_cols = ['Hours_in_state']

# zscore_cols = ['psi1', 'psi2', 'psi3', 'psi4', 'psi5']
# zscore_cols=['Temperature', "psi1", "psi2", "psi3", "psi4"]
zscore_cols = ['Temperature', 'Precipitation', 'Snow_depth', 'psi6', 'psi7']
# zscore_cols = ['Temperature', 'psi6', 'psi7']

# zscore_cols = []



# t_start = time()

# transition_models, train_datasets, test_datasets, ess_res, scalers = gam_models.train_all_region_transition_models( all_data_df= all_data_df,
#                                                                             regions= states_list[:1],
#                                                                             feature_cols= feature_cols,
#                                                                             zscore_cols=zscore_cols,
#                                                                             regional_classifier_features= ['Temperature', 'Relative_humidity', 'Load_CDF', 'Temperature_3Dsum_hot', 'Temperature_3Dsum_cold', 'month_sin', 'month_cos'],
#                                                                             type_model='GAM',
#                                                                             test_frac= 0.2,
#                                                                             specific_test_periods= specific_test_periods_per_state,
#                                                                             seed= 42,
#                                                                             w_region_consider= False,
#                                                                             w_stress_consider= False,
#                                                                             gamma= gamma,
#                                                                             clipping_quantile= 0.95,
#                                                                             verbose= False)

# t_end = time()

# print(f"Transition model successfully trained in {t_end - t_start:.2f} seconds. Started at {pd.Timestamp(t_start, unit='s')}, ended at {pd.Timestamp(t_end, unit='s')}")
# print("Effective sample size results:", ess_res[states_list[0]], ' = ', ess_res[states_list[0]]/len(list(train_datasets.values())[0]), ' %')

# gam_models.export_transition_model_bundle(
#     THIS_DIR/f"../Results/GAM/{folder_name}/{model_name}{spline_version}/model_bundle_gam.joblib",
#     transition_models_by_region=transition_models,
#     scalers_by_region=scalers,
#     feature_cols=feature_cols,
#     zscore_cols=zscore_cols,
#     model_type="GAM",
#     extra_metadata={"gamma": gamma, "clipping_quantile": 0.95},
# )

# print("Transition model bundle successfully exported")

# gam_models.export_gam_predictions(
#     transition_models= transition_models,
#     test_datasets= test_datasets,
#     scalers_by_region= scalers,
#     cols_export=['Datetime_UTC', 'State', 'Stress', 'Initial_gen_state', 'Final_gen_state', 'Data_weight', 'pAD', 'pAO', 'pDA', 'pOA']+feature_cols,
#     region_only=False,
#     test=True,
#     feature_cols= feature_cols,
#     out_dir=THIS_DIR/f'../Results/GAM/{folder_name}/{model_name}{spline_version}/',
#     model_name=model_name
# )

# print("Transition model successfully exported")



t_start = time()
transition_models_logistic, train_datasets_logistic, test_datasets_logistic, ess_res_logistic, scalers_logistic = gam_models.train_all_region_transition_models( all_data_df= all_data_df,
                                                                            regions= states_list[:1],
                                                                            feature_cols= feature_cols,
                                                                            zscore_cols=zscore_cols,
                                                                            regional_classifier_features= ['Temperature', 'Relative_humidity', 'Load_CDF', 'Temperature_3Dsum_hot', 'Temperature_3Dsum_cold', 'month_sin', 'month_cos'],
                                                                            # base_model_factory= base_model_factory,
                                                                            type_model='LogisticRegression',
                                                                            test_frac= 0.2,
                                                                            specific_test_periods= specific_test_periods_per_state,
                                                                            seed= 42,
                                                                            w_region_consider= False,
                                                                            w_stress_consider= True,
                                                                            gamma= gamma,
                                                                            clipping_quantile= 0.95,
                                                                            verbose= False)

t_end = time()
print(f"Transition model successfully trained in {t_end - t_start:.2f} seconds. Started at {pd.Timestamp(t_start, unit='s')}, ended at {pd.Timestamp(t_end, unit='s')}")


gam_models.export_transition_model_bundle(
    THIS_DIR/f"../Results/GAM/{folder_name}/{model_name}{spline_version}/model_bundle_gam.joblib",
    transition_models_by_region=transition_models_logistic,
    scalers_by_region=scalers_logistic,
    feature_cols=feature_cols,
    zscore_cols=zscore_cols,
    model_type="LogisticReg",
    extra_metadata={"gamma": gamma, "clipping_quantile": 0.95},
)

print("Transition model bundle successfully exported")

gam_models.export_gam_predictions(
    transition_models= transition_models_logistic,
    test_datasets= test_datasets_logistic,
    scalers_by_region= scalers_logistic,
    cols_export=['Datetime_UTC', 'State', 'Stress', 'Initial_gen_state', 'Final_gen_state', 'Data_weight', 'pAD', 'pAO', 'pDA', 'pOA']+feature_cols,
    region_only=False,
    test=True,
    feature_cols= feature_cols,
    out_dir=THIS_DIR/f"../Results/GAM/{folder_name}/{model_name}{spline_version}/",
    model_name='LogisticReg'
)

print("Transition model successfully exported")