import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV

def initial_hyperparam_tuning(X, y, seed=4):
  param_grid = {
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True]
  }
  search = RandomizedSearchCV(
      estimator=RandomForestClassifier(
        n_estimators=500, 
        random_state=seed, 
        n_jobs=1
      ), 
      param_distributions=param_grid, 
      n_iter=15, 
      scoring='f1',
      cv=3, 
      n_jobs=1,
      random_state=seed
  )
  search.fit(X, y)
  return search.best_params_

def train_random_forest(X_train, y_train, params, seed):
  '''
    Trains a sklearn RandomForestClassifier on the given training data with the provided best parameters

    Inputs:
      X_train: training features
      y_train: training labels

    Outputs:
      rf: trained RandomForestClassifier
  '''

  if params is None:
    params = {
      "max_depth": 10,
      "max_features": "sqrt",
      "min_samples_split": 5,
      "min_samples_leaf": 2
    }

  #create the RF classifier
  rf = RandomForestClassifier(
    **params, 
    n_estimators=500, 
    random_state=seed, 
    n_jobs=1
  )
  
  rf.fit(X_train, y_train)

  return rf

def find_threshold_at_target_ppv(y_true, y_probs, target_ppv, epsilon=0.05):
  """
    finds the classification threshold to achieve the target PPV across the predictions
  """
  y_true_arr = np.asarray(y_true)
  y_probs_arr = np.asarray(y_probs)

  precisions, _, thresholds = precision_recall_curve(y_true_arr, y_probs_arr)

  # find the lowest threshold for which the precision is over the target
  valid_indices = np.where(precisions >= target_ppv)[0]
  valid_indices = valid_indices[valid_indices < len(thresholds)]

  # If no valid indice is found, return the highest threshold
  if len(valid_indices) == 0:
    return float(thresholds[-1])

  for idx in valid_indices:
    current_tau = thresholds[idx]
    
    # Find the index where the threshold has increased by exactly epsilon
    future_indices = np.where(thresholds <= (current_tau + epsilon))[0]
    if len(future_indices) == 0:
      continue
    end_idx = future_indices[-1]
        
    # Extract the minimum precision across this physical probability span
    window_min_precision = np.min(precisions[idx:end_idx])
    
    if window_min_precision >= (target_ppv * 0.98):
      return float(current_tau)
          
  return float(thresholds[valid_indices[0]])
  

def prepare_datasets(
  patient_indices: np.ndarray | list[int],
  base_df: pd.DataFrame, 
  latents_df: pd.DataFrame, 
  cf_df: pd.DataFrame, 
  feature_mapping: dict
) -> dict[str, dict[str, pd.Series | np.ndarray]]:
  """
  Constructs aligned feature matrices (X, y, X_cf) for all 5 model variants.
  Can be applied uniformly to Sub-Training, Sub-Validation, or Pipeline Test splits.
  """

  x_desc_cols = [f['name'] for f in feature_mapping['desc']]
  x_indcorr_cols = [f['name'] for f in feature_mapping['indcorr']]
  x_sens_col = [f['name'] for f in feature_mapping['sens']][0]
  target_col = feature_mapping['target']['name']
  u_desc_cols = latents_df.filter(regex="u_desc_.*").columns.to_list()

  idx_list = list(patient_indices)
  sub_base = base_df.loc[base_df.index.isin(idx_list)].copy()
  sub_latents = latents_df[latents_df['patient_index'].isin(idx_list)].copy()
  sub_cf = cf_df[cf_df['patient_index'].isin(idx_list)].copy()

  merged_fair = sub_latents.merge(
    sub_base[x_indcorr_cols + [x_sens_col, target_col]], 
    right_index=True, 
    left_on="patient_index"
  )
  fair_cf = merged_fair.copy()
  fair_cf[x_sens_col] = 1 - fair_cf[x_sens_col]

  merged_cf = sub_cf.merge(
    sub_base[x_indcorr_cols + [x_sens_col, target_col]], 
    left_on="patient_index", 
    right_index=True,
    suffixes=("", "_fact")
  )
  merged_cf[x_sens_col] = 1 - merged_cf[x_sens_col] 

  datasets = {}

  baseline_features = x_indcorr_cols + x_desc_cols + [x_sens_col]
  baseline_unaware_features = x_indcorr_cols + x_desc_cols
  ablation_features = x_indcorr_cols
  fair_features = x_indcorr_cols + u_desc_cols + [x_sens_col]
  fair_unaware_features = x_indcorr_cols + u_desc_cols

  # CF Patient indices for alignment in result processing
  cf_patient_arr = merged_cf["patient_index"].to_numpy()
  cf_sens_arr = merged_cf[x_sens_col].to_numpy()
  cf_fair_sens_arr = fair_cf[x_sens_col].to_numpy()

  datasets["baseline"] = {
    "X": sub_base[baseline_features],
    "y": sub_base[target_col],
    "X_cf": merged_cf[baseline_features],
    "sens": sub_base[x_sens_col].to_numpy(),
    "patient_index": sub_base.index.to_numpy(),
    "cf_sens": cf_sens_arr,
    "cf_patient_index": cf_patient_arr
  }

  datasets["baseline_unaware"] = {
    "X": sub_base[baseline_unaware_features],
    "y": sub_base[target_col],
    "X_cf": merged_cf[baseline_unaware_features],
    "sens": sub_base[x_sens_col].to_numpy(),
    "patient_index": sub_base.index.to_numpy(),
    "cf_sens": cf_sens_arr,
    "cf_patient_index": cf_patient_arr
  }

  datasets["ablation"] = {
    "X": sub_base[ablation_features].copy(),
    "y": sub_base[target_col].copy(),
    "X_cf": merged_cf[ablation_features],
    "sens": sub_base[x_sens_col].to_numpy(),
    "patient_index": sub_base.index.to_numpy(),
    "cf_sens": cf_sens_arr,
    "cf_patient_index": cf_patient_arr
  }

  datasets["fair"] = {
    "X": merged_fair[fair_features].copy(),
    "y": merged_fair[target_col].copy(),
    "X_cf": fair_cf[fair_features].copy(),
    "sens": merged_fair[x_sens_col].to_numpy(),
    "patient_index": merged_fair["patient_index"].to_numpy(),
    "cf_sens": cf_fair_sens_arr,
    "cf_patient_index": fair_cf["patient_index"].to_numpy()
  }

  datasets["fair_unaware"] = {
    "X": merged_fair[fair_unaware_features].copy(),
    "y": merged_fair[target_col].copy(),
    "X_cf": fair_cf[fair_unaware_features].copy(),
    "sens": merged_fair[x_sens_col].to_numpy(),
    "patient_index": merged_fair["patient_index"].to_numpy(),
    "cf_sens": cf_fair_sens_arr,
    "cf_patient_index": fair_cf["patient_index"].to_numpy()
  }

  return datasets