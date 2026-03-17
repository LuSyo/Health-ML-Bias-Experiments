import argparse
import os
import gc

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config
from cevaehe.model import CEVAEHE
from classifiers.train import initial_hyperparam_tuning, train_random_forest
from utils import parse_args, load_feature_mapping, set_global_seeds, setup_logger
from metrics import calculate_performance_metrics, get_grouped_roc_curve, stratified_perf, avg_perf_per_patient,get_interp_tpr
from plots import stratified_roc_curves

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
    # cevaehe = CEVAEHE(feature_mapping['ind'], 
    #                 feature_mapping['desc'], 
    #                 feature_mapping['corr'], 
    #                 feature_mapping['sens'], 
    #               args=args)
    # model_path = f'{Config.MODELS_DIR}{args.cevaehe}'
    # torch.serialization.add_safe_globals([argparse.Namespace])
    # model_state = torch.load(model_path, weights_only=True)
    # cevaehe.load_state_dict(model_state['model_state_dict']) 


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
    # X = dataset.drop([target], axis=1)
    X = dataset[feature_cols]
    X[x_sens_col[0]] = dataset[x_sens_col[0]]
    y = dataset[target]

    # Sociological counterfactual
    # X_soc_cf = counterfactuals_df[x_desc_cols].merge(dataset[x_ind_cols + x_corr_cols],
    #                                                          left_index=True,
    #                                                          right_index=True)[feature_cols]
    # X_soc_cf['s_bio'] = dataset[x_sens_col[0]]
    # X_soc_cf['s_soc'] = 1 - dataset[x_sens_col[0]]

    # Merge latents, Xsens and Xind into fair features dataframe
    fair_dataset = latents_df.merge(dataset[feature_cols + x_sens_col + [target]], right_index=True, left_on='patient_index')

    sss = StratifiedShuffleSplit(n_splits=args.n_runs, test_size=0.3, random_state=42)

    baseline_metrics = []
    fair_0_metrics = []
    fair_1_metrics = []
    fair_2_metrics = []
    fair_3_metrics = []
    baseline_feature_importances = []
    feature_names = X.columns.tolist()

    sens_groups = np.unique(X[x_sens_col[0]].values)

    roc_curve_data = {
      model: {group: [] for group in sens_groups} 
      for model in ['baseline', 'fair_0', 'fair_1', 'fair_2', 'fair_3']
    }

    logger.info("Performing initial hyperparameter tuning...")
    best_params = initial_hyperparam_tuning(X, y)
    logger.info(f"Best Params found: {best_params}")

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
      fair_0_X_train = fair_dataset.loc[fair_train_index, fair_0_cols].copy()
      fair_0_X_test = fair_dataset.loc[fair_test_index, fair_0_cols].copy()

      fair_1_cols = x_desc_cols + u_corr_cols + x_ind_cols
      if i == 0: logger.info(f'Model 1 columns: {fair_1_cols}')
      fair_1_X_train = fair_dataset.loc[fair_train_index, fair_1_cols].copy()
      fair_1_X_test = fair_dataset.loc[fair_test_index, fair_1_cols].copy()

      fair_2_cols = u_desc_cols + x_corr_cols + x_ind_cols
      if i == 0: logger.info(f'Model 2 columns: {fair_2_cols}')
      fair_2_X_train = fair_dataset.loc[fair_train_index, fair_2_cols].copy()
      fair_2_X_test = fair_dataset.loc[fair_test_index, fair_2_cols].copy()

      fair_3_cols = u_desc_cols + u_corr_cols
      if i == 0: logger.info(f'Model 3 columns: {fair_3_cols}')
      fair_3_X_train = fair_dataset.loc[fair_train_index, fair_3_cols].copy()
      fair_3_X_test = fair_dataset.loc[fair_test_index, fair_3_cols].copy()

      fair_y_train = fair_dataset.loc[fair_train_index, target].copy()
      fair_y_test = fair_dataset.loc[fair_test_index, target].copy()
      fair_s_ref = fair_dataset.loc[fair_test_index, x_sens_col[0]].copy()

      # Train the baseline and fair models
      logger.info("Train baseline model")
      rf_baseline, y_pred, y_pred_proba = train_random_forest(X_train, y_train, X_test, best_params, args.seed)

      importances = rf_baseline.feature_importances_
      baseline_feature_importances.append(importances)

      del rf_baseline
      
      logger.info("Train Fair Model 0: Ucorr, Udesc, Xind")
      _, fair_0_y_pred, fair_0_y_pred_proba = train_random_forest(
      fair_0_X_train, fair_y_train, fair_0_X_test, best_params, args.seed)

      del _
      
      logger.info("Train Fair Model 1: Ucorr, Xdesc, Xind")
      _, fair_1_y_pred, fair_1_y_pred_proba = train_random_forest(
      fair_1_X_train, fair_y_train, fair_1_X_test, best_params, args.seed)

      del _
      
      logger.info("Train Fair Model 2: Xcorr, Udesc, Xind")
      _, fair_2_y_pred, fair_2_y_pred_proba = train_random_forest(
      fair_2_X_train, fair_y_train, fair_2_X_test, best_params, args.seed)

      del _
      
      logger.info("Train Fair Model 3: Ucorr, Udesc")
      _, fair_3_y_pred, fair_3_y_pred_proba = train_random_forest(
      fair_3_X_train, fair_y_train, fair_3_X_test, best_params, args.seed)

      del _

      # TODO: run the CEVAEHE on the test set

      #BASELINE PERF METRICS
      baseline_global_perf = calculate_performance_metrics(y_test, y_pred, y_pred_proba)
      baseline_strat_perf = stratified_perf(y_test, y_pred, y_pred_proba, X_test[x_sens_col[0]])
      baseline_metrics.append(baseline_global_perf | baseline_strat_perf)

      # FAIR MODELS PERF METRICS
      patient_indexes = fair_dataset.loc[fair_test_index, 'patient_index'].values

      fair_0_global_perf, fair_0_strat_perf, fair_0_roc_curves = avg_perf_per_patient(fair_y_test, fair_0_y_pred_proba, fair_s_ref.values, patient_indexes)
      fair_0_metrics.append(fair_0_global_perf | fair_0_strat_perf)

      fair_1_global_perf, fair_1_strat_perf, fair_1_roc_curves = avg_perf_per_patient(fair_y_test, fair_1_y_pred_proba, fair_s_ref.values, patient_indexes)
      fair_1_metrics.append(fair_1_global_perf | fair_1_strat_perf)

      fair_2_global_perf, fair_2_strat_perf, fair_2_roc_curves = avg_perf_per_patient(fair_y_test, fair_2_y_pred_proba, fair_s_ref.values, patient_indexes)
      fair_2_metrics.append(fair_2_global_perf | fair_2_strat_perf)

      fair_3_global_perf, fair_3_strat_perf, fair_3_roc_curves = avg_perf_per_patient(fair_y_test, fair_3_y_pred_proba, fair_s_ref.values, patient_indexes)
      fair_3_metrics.append(fair_3_global_perf | fair_3_strat_perf)

      for name, curves in [
        ('baseline', get_grouped_roc_curve(y_test, y_pred_proba, X_test[x_sens_col[0]])),
        ('fair_0', fair_0_roc_curves),
        ('fair_1', fair_1_roc_curves),
        ('fair_2', fair_2_roc_curves),
        ('fair_3', fair_3_roc_curves)
      ]:
        for group_id, tpr_interp in curves.items():
            roc_curve_data[name][group_id].append(tpr_interp)

      # eval_configs = [
      #     ('baseline', y_test, y_pred_proba, X_test[x_sens_col[0]]),
      #     ('fair_0', fair_y_test, fair_0_y_pred_proba, fair_s_ref.values),
      #     ('fair_1', fair_y_test, fair_1_y_pred_proba, fair_s_ref.values),
      #     ('fair_2', fair_y_test, fair_2_y_pred_proba, fair_s_ref.values),
      #     ('fair_3', fair_y_test, fair_3_y_pred_proba, fair_s_ref.values),
      # ]

      # for name, y_t, y_p, s_ref in eval_configs:
      #   for group in [0, 1]:
      #     mask = (s_ref == group)
      #     # Calculate interpolated TPR for this run/group
      #     tpr_interp = get_interp_tpr(y_t[mask], y_p[mask])
      #     roc_curve_data[name][group].append(tpr_interp)

      # print(f"Run {i}: Test set size: {len(y_test)}, Females: {np.sum(X_test['s_bio']==0)}, Female Positives: {np.sum(y_test[X_test['s_bio']==0])}")
      # print(f"Run {i}: Test set size: {len(y_test)}, Males: {np.sum(X_test['s_bio']==1)}, Male Positives: {np.sum(y_test[X_test['s_bio']==1])}")

    # --- SAVE PERFORMANCE METRICS ---
    logger.info('Saving results and cleaning up memory...')

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

    # --- SAVE STRATIFIED ROC CURVES ---

    final_curves = {'mean_fpr': np.linspace(0, 1, 100)}

    for model_name, groups in roc_curve_data.items():
        for group_id in sens_groups:
            mean_tpr = np.mean(groups[group_id], axis=0)
            mean_tpr[-1] = 1.0 
            
            final_curves[f"{model_name}_group_{group_id}"] = mean_tpr

    roc_curves_fig = stratified_roc_curves(final_curves)
    roc_curves_fig.savefig(f'{results_path}/roc_curves.png', bbox_inches='tight', dpi=300)

    # --- SAVE BASELINE FEATURE IMPORTANCES ---

    avg_importances = np.mean(baseline_feature_importances, axis=0)
    std_importances = np.std(baseline_feature_importances, axis=0)

    importance_df = pd.DataFrame({
      'feature': feature_names,
      'importance_mean': avg_importances,
      'importance_std': std_importances
    }).sort_values(by='importance_mean', ascending=False)

    importance_df.to_csv(f'{results_path}/baseline_feature_importances.csv', index=False)

    del baseline_metrics_df, fair_0_metrics_df, fair_1_metrics_df, fair_2_metrics_df, fair_3_metrics_df
    del final_curves
    del importance_df

    gc.collect()

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)

if __name__ == "__main__":
  main()

