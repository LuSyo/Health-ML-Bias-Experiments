import argparse
import os

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config
from cevaehe.model import CEVAEHE
from classifiers.train import train_random_forest
from utils import parse_args, load_feature_mapping, set_global_seeds, setup_logger
from metrics import calculate_performance_metrics, stratified_perf

def main():
  args = parse_args()

  set_global_seeds(args.seed)

  results_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(results_path, exist_ok=True)

  assert args.cevaehe is not None, "Missing CEVAEHE file name"
  assert args.cf_dataset is not None, "Missing counterfactual dataset file path"
  assert args.latent_dataset is not None, "Missing latent spaces dataset file path"

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)
  logger.info(f'EXPERIMENT START: {args.exp_name}')
  logger.info(f'Data: {args.data}')
  logger.info(f'Mapping: {args.mapping}')
  logger.info(f'CEVAEHE Model: {args.cevaehe}')
  logger.info(f'Counterfactual dataset: {args.cf_dataset}')
  logger.info(f'Latent spaces dataset: {args.latent_dataset}')

  try:
    # Load the feature mapping
    feature_mapping = load_feature_mapping(args.mapping)

    # Load the dataset
    dataset = pd.read_csv(Config.DATA_DIR + args.data)

    # Load the latent spaces dataset
    latents_df = pd.read_csv(Config.DATA_DIR + args.latent_dataset)

    # Load the counterfactual dataset
    counterfactuals_df = pd.read_csv(Config.DATA_DIR + args.cf_dataset)

    # Initialise CEVAEHE model and load with trained parameters
    cevaehe = CEVAEHE(feature_mapping['ind'], 
                    feature_mapping['desc'], 
                    feature_mapping['corr'], 
                    feature_mapping['sens'], 
                  args=args)
    model_path = f'{Config.MODELS_DIR}{args.cevaehe}'
    torch.serialization.add_safe_globals([argparse.Namespace])
    model_state = torch.load(model_path, weights_only=True)
    cevaehe.load_state_dict(model_state['model_state_dict']) 


    # Define feature columns to make sure that factual and coutnerfactual datasets
    # have the same column order
    x_ind_cols = [f['name'] for f in feature_mapping['ind']]
    x_desc_cols = [f['name'] for f in feature_mapping['desc']]
    x_corr_cols = [f['name'] for f in feature_mapping['corr']]
    x_sens_col = [f['name'] for f in feature_mapping['sens']]
    feature_cols = x_ind_cols + x_desc_cols + x_corr_cols
    target = feature_mapping['target']['name']
    u_desc_cols = latents_df.filter(regex="u_d_.*").columns.to_list()
    u_corr_cols = latents_df.filter(regex="u_c_.*").columns.to_list()

    # Baseline features and target class
    X = dataset[feature_cols]
    X['s_bio'] = dataset[x_sens_col[0]]
    X['s_soc'] = dataset[x_sens_col[0]]
    y = dataset[target]

    # Sociological counterfactual
    X_soc_cf = counterfactuals_df[x_desc_cols].merge(dataset[x_ind_cols + x_corr_cols],
                                                             left_index=True,
                                                             right_index=True)[feature_cols]
    X_soc_cf['s_bio'] = dataset[x_sens_col[0]]
    X_soc_cf['s_soc'] = 1 - dataset[x_sens_col[0]]

    # Merge latents, Xsens and Xind into fair features dataframe
    fair_dataset = latents_df.merge(dataset[feature_cols + x_sens_col + [target]], right_index=True, left_on='patient_index')

    sss = StratifiedShuffleSplit(n_splits=args.n_runs, test_size=0.2, random_state=42)

    baseline_metrics = []
    fair_0_metrics = []
    fair_1_metrics = []
    fair_2_metrics = []
    fair_3_metrics = []

    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
      logger.info(f'--- Start bootstrap loop {i}')
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      # Equivalent fair training and test feature sets for each fair model
      # Model 0: Ucorr, Udesc, Xind
      # Model 1: Ucorr, Xdesc, Xind
      # Model 2: Xcorr, Udesc, Xind
      # Model 3: Ucorr, Udesc
      fair_train_index = fair_dataset['patient_index'].isin(train_index)
      fair_test_index = fair_dataset['patient_index'].isin(test_index)

      fair_0_cols = u_desc_cols + u_corr_cols + x_ind_cols
      if i == 0: logger.info(f'Model 0 columns: {fair_0_cols}')
      fair_0_X_train = fair_dataset.loc[fair_train_index, fair_0_cols]
      fair_0_X_test = fair_dataset.loc[fair_test_index, fair_0_cols]

      fair_1_cols = x_desc_cols + u_corr_cols + x_ind_cols
      if i == 0: logger.info(f'Model 1 columns: {fair_1_cols}')
      fair_1_X_train = fair_dataset.loc[fair_train_index, fair_1_cols]
      fair_1_X_test = fair_dataset.loc[fair_test_index, fair_1_cols]

      fair_2_cols = u_desc_cols + x_corr_cols + x_ind_cols
      if i == 0: logger.info(f'Model 2 columns: {fair_2_cols}')
      fair_2_X_train = fair_dataset.loc[fair_train_index, fair_2_cols]
      fair_2_X_test = fair_dataset.loc[fair_test_index, fair_2_cols]

      fair_3_cols = u_desc_cols + u_corr_cols
      if i == 0: logger.info(f'Model 3 columns: {fair_3_cols}')
      fair_3_X_train = fair_dataset.loc[fair_train_index, fair_3_cols]
      fair_3_X_test = fair_dataset.loc[fair_test_index, fair_3_cols]

      fair_y_train = fair_dataset.loc[fair_train_index, target]
      fair_y_test = fair_dataset.loc[fair_test_index, target]
      fair_s_ref = fair_dataset.loc[fair_test_index, x_sens_col[0]]

      # Train the baseline and fair models
      logger.info("Train baseline model")
      _, y_pred, y_pred_proba = train_random_forest(X_train, y_train, X_test)
      
      logger.info("Train Fair Model 0: Ucorr, Udesc, Xind")
      _, fair_0_y_pred, fair_0_y_pred_proba = train_random_forest(
      fair_0_X_train, fair_y_train, fair_0_X_test)
      
      logger.info("Train Fair Model 1: Ucorr, Xdesc, Xind")
      _, fair_1_y_pred, fair_1_y_pred_proba = train_random_forest(
      fair_1_X_train, fair_y_train, fair_1_X_test)
      
      logger.info("Train Fair Model 2: Xcorr, Udesc, Xind")
      _, fair_2_y_pred, fair_2_y_pred_proba = train_random_forest(
      fair_2_X_train, fair_y_train, fair_2_X_test)
      
      logger.info("Train Fair Model 3: Ucorr, Udesc")
      _, fair_3_y_pred, fair_3_y_pred_proba = train_random_forest(
      fair_3_X_train, fair_y_train, fair_3_X_test)

      # TODO: run the CEVAEHE on the test set

      #GLOBAL PERFORMANCE METRICS
      baseline_global_perf = calculate_performance_metrics(y_test, y_pred, y_pred_proba)
      fair_0_global_perf = calculate_performance_metrics(fair_y_test, fair_0_y_pred, fair_0_y_pred_proba)
      fair_1_global_perf = calculate_performance_metrics(fair_y_test, fair_1_y_pred, fair_1_y_pred_proba)
      fair_2_global_perf = calculate_performance_metrics(fair_y_test, fair_2_y_pred, fair_2_y_pred_proba)
      fair_3_global_perf = calculate_performance_metrics(fair_y_test, fair_3_y_pred, fair_3_y_pred_proba)

      # STRATIFIED PERFORMANCE METRICS
      baseline_strat_perf = stratified_perf(y_test, y_pred, y_pred_proba, X_test['s_bio'])
      fair_0_strat_perf = stratified_perf(fair_y_test, fair_0_y_pred, fair_0_y_pred_proba, fair_s_ref.values)
      fair_1_strat_perf = stratified_perf(fair_y_test, fair_1_y_pred, fair_1_y_pred_proba, fair_s_ref.values)
      fair_2_strat_perf = stratified_perf(fair_y_test, fair_2_y_pred, fair_2_y_pred_proba, fair_s_ref.values)
      fair_3_strat_perf = stratified_perf(fair_y_test, fair_3_y_pred, fair_3_y_pred_proba, fair_s_ref.values)

      baseline_metrics.append(baseline_global_perf | baseline_strat_perf)
      fair_0_metrics.append(fair_0_global_perf | fair_0_strat_perf)
      fair_1_metrics.append(fair_1_global_perf | fair_1_strat_perf)
      fair_2_metrics.append(fair_2_global_perf | fair_2_strat_perf)
      fair_3_metrics.append(fair_3_global_perf | fair_3_strat_perf)

    baseline_metrics_df = pd.DataFrame(baseline_metrics) 
    fair_0_metrics_df = pd.DataFrame(fair_0_metrics) 
    fair_1_metrics_df = pd.DataFrame(fair_1_metrics) 
    fair_2_metrics_df = pd.DataFrame(fair_2_metrics) 
    fair_3_metrics_df = pd.DataFrame(fair_3_metrics) 

    baseline_metrics_df.to_csv(f'{results_path}/baseline_metrics.csv', index=False)
    fair_0_metrics_df.to_csv(f'{results_path}/fair_0_metrics.csv', index=False)
    fair_1_metrics_df.to_csv(f'{results_path}/fair_1_metrics.csv', index=False)
    fair_2_metrics_df.to_csv(f'{results_path}/fair_2_metrics.csv', index=False)
    fair_3_metrics_df.to_csv(f'{results_path}/fair_3_metrics.csv', index=False)

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)

if __name__ == "__main__":
  main()