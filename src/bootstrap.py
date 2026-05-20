import os
import gc

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config
from cevaehe.model import CEVAEHE
from classifiers.train import initial_hyperparam_tuning, train_random_forest
from utils import parse_args, load_config, set_global_seeds, setup_logger
from metrics import calculate_performance_metrics, get_grouped_roc_curve, stratified_perf, avg_perf_per_patient, calculate_counterfactual_harm
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
    feature_mapping = load_config(args.mapping)

    # Load the dataset
    dataset = pd.read_csv(Config.DATA_DIR + args.data)

    # Load the latent spaces dataset
    latents_df = pd.read_csv(Config.DATA_DIR + args.latent_dataset)

    # Load the counterfactual dataset
    counterfactuals_df = pd.read_csv(Config.DATA_DIR + args.cf_dataset)

    # Define the feature columns
    x_desc_cols = [f['name'] for f in feature_mapping['desc']]
    x_indcorr_cols = [f['name'] for f in feature_mapping['indcorr']]
    x_sens_col = [f['name'] for f in feature_mapping['sens']][0]
    feature_cols = x_indcorr_cols + x_desc_cols
    target = feature_mapping['target']['name']
    u_desc_cols = latents_df.filter(regex="u_desc_.*").columns.to_list()

    baseline_features = x_indcorr_cols + x_desc_cols + [x_sens_col]
    baseline_unaware_features = x_indcorr_cols + x_desc_cols
    fair_features = x_indcorr_cols + u_desc_cols + [x_sens_col]
    fair_unaware_features = x_indcorr_cols + u_desc_cols
    logger.info(f'Baseline Model columns: {baseline_features}')
    logger.info(f'Baseline S-unaware Model columns: {baseline_unaware_features}')
    logger.info(f'Fair Model columns: {fair_features}')
    logger.info(f'Fair S-unaware Model columns: {fair_unaware_features}')
    
    # Fair dataset
    fair_dataset = latents_df.merge(
      dataset[x_indcorr_cols + [x_sens_col, target]], 
      right_index=True, 
      left_on="patient_index")

    # Counterfactual dataset
    cf_dataset = counterfactuals_df.merge(
      dataset[x_indcorr_cols + [x_sens_col, target]],
      right_index=True,
      left_on="patient_index",
      suffixes=("_fact", "")
    )
    cf_dataset[x_sens_col] = 1 - cf_dataset[x_sens_col] 

    # Baseline features and targets
    X = dataset[baseline_features].copy()
    y = dataset[target].copy()

    sss = StratifiedShuffleSplit(n_splits=args.n_runs, test_size=0.3, random_state=args.seed)

    baseline_metrics = []
    baseline_unaware_metrics = []
    fair_metrics = []
    fair_unaware_metrics = []

    baseline_feat_imp = []
    baseline_unaware_feat_imp = []
    fair_feat_imp = []
    fair_unaware_feat_imp = []

    sens_groups = X.loc[:, x_sens_col].unique()

    roc_curve_data = {
      model: {group: [] for group in sens_groups} 
      for model in ['baseline', 'baseline_unaware', 'fair', 'fair_unaware']
    }

    logger.info("Performing initial hyperparameter tuning...")
    best_params = initial_hyperparam_tuning(X, y)
    logger.info(f"Best Params found: {best_params}")

    # print(fair_dataset.head(5).to_markdown())
    # print(cf_dataset.head(5).to_markdown())

    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
      logger.info(f'--- Start bootstrap loop {i}')
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      X_unaware_train = X_train.drop(x_sens_col, axis=1)
      X_unaware_test = X_test.drop(x_sens_col, axis=1)
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      # Fair training features and target
      fair_train_index = fair_dataset['patient_index'].isin(train_index)
      fair_test_index = fair_dataset['patient_index'].isin(test_index)

      fair_X_train = fair_dataset.loc[fair_train_index, fair_features].copy()
      fair_X_test = fair_dataset.loc[fair_test_index, fair_features].copy()
      fair_unaware_X_train = fair_X_train.drop(x_sens_col, axis=1)
      fair_unaware_X_test = fair_X_test.drop(x_sens_col, axis=1)

      fair_y_train = fair_dataset.loc[fair_train_index, target].copy()
      fair_y_test = fair_dataset.loc[fair_test_index, target].copy()

      # TEST ONLY: Counterfactual features and target
      cf_test_index = cf_dataset['patient_index'].isin(test_index)

      cf_X_test = cf_dataset.loc[cf_test_index, baseline_features].copy()
      cf_X_unaware_test = cf_X_test.drop(x_sens_col, axis=1)
      
      cf_fair_X_test = fair_X_test.copy()
      cf_fair_X_test[x_sens_col] = 1 - cf_fair_X_test[x_sens_col] 

      # --- TRAIN MODELS ---
      logger.info("Train baseline model")
      rf_baseline, y_pred, y_pred_proba, y_cf_pred_proba, baseline_threshold = train_random_forest(
        X_train, y_train, X_test, cf_X_test, 
        best_params, args.target_ppv, args.seed)

      baseline_feat_imp.append(rf_baseline.feature_importances_)

      del rf_baseline

      logger.info("Train baseline S-unaware model")
      rf_baseline_unaware, un_y_pred, un_y_pred_proba, un_y_cf_pred_proba, baseline_unaware_threshold = train_random_forest(
        X_unaware_train, y_train, X_unaware_test, cf_X_unaware_test, 
        best_params, args.target_ppv, args.seed)

      baseline_unaware_feat_imp.append(rf_baseline_unaware.feature_importances_)

      del rf_baseline_unaware

      logger.info("Train fair model")
      rf_fair, fair_y_pred, fair_y_pred_proba, fair_y_cf_pred_proba, fair_threshold = train_random_forest(
        fair_X_train, fair_y_train, fair_X_test, cf_fair_X_test, 
        best_params, args.target_ppv, args.seed)

      fair_feat_imp.append(rf_fair.feature_importances_)

      del rf_fair

      logger.info("Train fair S-unaware model")
      rf_fair_unaware, fair_un_y_pred, fair_un_y_pred_proba, fair_un_y_cf_pred_proba, fair_unaware_threshold = train_random_forest(
        fair_unaware_X_train, fair_y_train, fair_unaware_X_test, fair_unaware_X_test, 
        best_params, args.target_ppv, args.seed)

      fair_unaware_feat_imp.append(rf_fair_unaware.feature_importances_)

      del rf_fair_unaware

      # --- TEST PERF METRICS ---

      # Baseline and Baseline S-unaware metrics
      def get_baseline_metrics(y_test, y_pred, y_pred_proba, y_cf_pred_proba, X_test_sens, threshold):
        global_perf = calculate_performance_metrics(y_test, y_pred, y_pred_proba)
        global_perf['threshold'] = threshold
        # global_perf['cf_harm'], global_perf['cf_harm_pos'], global_perf['cf_harm_neg'] = calculate_counterfactual_harm(
        #   y_test,
        #   y_pred_proba,
        #   y_cf_pred_proba
        # )

        strat_perf = stratified_perf(
          y_test, y_pred, y_pred_proba, 
          X_test_sens, 
          y_cf_pred_proba)

        return global_perf | strat_perf

      baseline_metrics.append(get_baseline_metrics(
        y_test=y_test, 
        y_pred=y_pred, 
        y_pred_proba=y_pred_proba, 
        y_cf_pred_proba=y_cf_pred_proba, 
        X_test_sens=X_test[x_sens_col], 
        threshold=baseline_threshold
      ))

      baseline_unaware_metrics.append(get_baseline_metrics(
        y_test=y_test, 
        y_pred=un_y_pred, 
        y_pred_proba=un_y_pred_proba, 
        y_cf_pred_proba=un_y_cf_pred_proba, 
        X_test_sens=X_test[x_sens_col], 
        threshold=baseline_unaware_threshold
      ))

      # Fair and Fair S-unaware metrics
      patient_indexes = fair_dataset.loc[fair_test_index, 'patient_index'].values

      fair_global_perf, fair_strat_perf, fair_roc_curves = avg_perf_per_patient(
        y_true=fair_y_test, 
        y_pred_prob=fair_y_pred_proba, 
        y_cf_pred_prob=fair_y_cf_pred_proba, 
        sens=fair_X_test[x_sens_col], 
        patient_index=patient_indexes, 
        threshold=fair_threshold
      )
      fair_metrics.append(fair_global_perf | fair_strat_perf)

      fair_un_global_perf, fair_un_strat_perf, fair_un_roc_curves = avg_perf_per_patient(
        y_true=fair_y_test, 
        y_pred_prob=fair_un_y_pred_proba, 
        y_cf_pred_prob=fair_un_y_cf_pred_proba, 
        sens=fair_X_test[x_sens_col], 
        patient_index=patient_indexes, 
        threshold=fair_unaware_threshold
      )
      fair_unaware_metrics.append(fair_un_global_perf | fair_un_strat_perf)

      for name, curves in [
        ('baseline', get_grouped_roc_curve(y_test, y_pred_proba, X_test[x_sens_col])),
        ('baseline_unaware', get_grouped_roc_curve(y_test, un_y_pred_proba, X_test[x_sens_col])),
        ('fair', fair_roc_curves),
        ('fair_unaware', fair_un_roc_curves)
      ]:
        for group_id, tpr_interp in curves.items():
            roc_curve_data[name][group_id].append(tpr_interp)

    logger.info('Saving results and cleaning up memory...')

    baseline_metrics_df = pd.DataFrame(baseline_metrics) 
    baseline_metrics_df.to_csv(f'{results_path}/baseline_metrics.csv', index=False)

    baseline_unaware_metrics_df = pd.DataFrame(baseline_unaware_metrics) 
    baseline_unaware_metrics_df.to_csv(f'{results_path}/baseline_unaware_metrics.csv', index=False)

    fair_metrics_df = pd.DataFrame(fair_metrics) 
    fair_metrics_df.to_csv(f'{results_path}/fair_metrics.csv', index=False)

    fair_unaware_metrics_df = pd.DataFrame(fair_unaware_metrics) 
    fair_unaware_metrics_df.to_csv(f'{results_path}/fair_unaware_metrics.csv', index=False)

    # --- SAVE STRATIFIED ROC CURVES ---

    final_curves = {'mean_fpr': np.linspace(0, 1, 100)}
    for model_name, groups in roc_curve_data.items():
      for group_id in sens_groups:
        mean_tpr = np.mean(groups[group_id], axis=0)
        mean_tpr[-1] = 1.0 
        
        final_curves[f"{model_name}_group_{group_id}"] = mean_tpr

    roc_curves_fig = stratified_roc_curves(
      final_curves,
      models = roc_curve_data.keys())
    roc_curves_fig.savefig(f'{results_path}/roc_curves.png', bbox_inches='tight', dpi=300)

    # --- SAVE FEATURE IMPORTANCES ---
    def save_feat_imp(feat_imp, feat_names, file_name):
      avg_importances = np.mean(feat_imp, axis=0)
      std_importances = np.std(feat_imp, axis=0)

      importance_df = pd.DataFrame({
        'feature': feat_names,
        'importance_mean': avg_importances,
        'importance_std': std_importances
      }).sort_values(by='importance_mean', ascending=False)

      importance_df.to_csv(f'{results_path}/{file_name}.csv', index=False)

    save_feat_imp(baseline_feat_imp, baseline_features, "baseline_feat_imp")
    save_feat_imp(baseline_unaware_feat_imp, baseline_unaware_features, "baseline_unaware_feat_imp")
    save_feat_imp(fair_feat_imp, fair_features, "fair_feat_imp")
    save_feat_imp(fair_unaware_feat_imp, fair_unaware_features, "fair_unaware_feat_imp")

    # del final_curves
    del baseline_metrics_df, baseline_unaware_metrics_df, fair_metrics_df, fair_unaware_metrics_df
    del baseline_metrics, baseline_unaware_metrics, fair_metrics, fair_unaware_metrics
    gc.collect()
    
  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)

if __name__ == "__main__":
  main()
