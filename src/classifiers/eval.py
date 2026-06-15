import pandas as pd
import numpy as np
from typing import Any, cast
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss, average_precision_score
from metrics import calculate_performance_metrics, calculate_counterfactual_harm

def eval_classifiers(
  classifiers: dict[str, Any], 
  test_datasets: dict[str, dict[str, Any]], 
  class_thresholds: dict[str, float],
  n_bootstraps: int = 1000,
  seed: int = 4
) -> pd.DataFrame:
  """
  Evaluates performance of stack of classifiers on the given test data through 1,000 bootstraps 

  Inputs:
    classifiers: dictionary of RF classifiers
    test_datasets: dictionary of test datasets
    class_thresholds: dictionary of classification thresholds 

  Outputs:
    metrics: dictionary of models' performance metrics   
  """

  rng = np.random.default_rng(seed)

  for model_key, rf in classifiers.items():
    dataset = test_datasets[model_key]
    tau = class_thresholds[model_key]

    y_prob = np.asarray(rf.predict_proba(dataset['X']))[:, 1]
    y_cf_prob = np.asarray(rf.predict_proba(dataset['X_cf']))[:, 1]

    # collapse standard and M-sampled datasets into one row per patient
    fact_df = cast(pd.DataFrame, pd.DataFrame({
      "patient_index": dataset["patient_index"],
      "y_true": np.asarray(dataset["y"]),
      "sens": dataset["sens"],
      "y_prob": y_prob
    }).groupby("patient_index").mean())

    cf_df = cast(pd.DataFrame, pd.DataFrame({
      "patient_index": dataset["cf_patient_index"],
      "y_cf_prob": y_cf_prob
    }).groupby("patient_index").mean())

    eval_df = fact_df.merge(cf_df, left_index=True, right_index=True).reset_index()
        
    n_patients = len(eval_df)

    # create list of patients' prediction results
    patient_records = eval_df.to_dict(orient="records")

    all_bootstrap_results = []

    for b in range(n_bootstraps):
      bootstrap_indices = rng.choice(n_patients, size=n_patients, replace=True)
      boot_samples = [patient_records[idx] for idx in bootstrap_indices]
      boot_df = pd.DataFrame(boot_samples)

      y_pred = (boot_df['y_prob'] >= tau).astype(int)
      y_true = boot_df["y_true"].to_numpy()

      y_prob_arr = boot_df['y_prob'].to_numpy()
      y_cf_prob_arr = boot_df['y_cf_prob'].to_numpy()

      global_perf_metrics = calculate_performance_metrics(y_true, y_pred, y_prob_arr)
      global_cf_harm = calculate_counterfactual_harm(y_true, y_prob_arr, y_cf_prob_arr)

      all_bootstrap_results.append({
        "model": model_key,
        "bootstrap_idx": b,
        "subgroup": "Global"
      } | global_perf_metrics | global_cf_harm)

      for group_id, group_df in boot_df.groupby("sens"):
        g_true = group_df["y_true"].to_numpy()
        g_prob = group_df["y_prob"].to_numpy()
        g_cf_prob = group_df["y_cf_prob"].to_numpy()
        g_pred = (g_prob >= tau).astype(int)

        group_perf_metrics = calculate_performance_metrics(g_true, g_pred, g_prob)
        group_cf_harm = calculate_counterfactual_harm(g_true, g_prob, g_cf_prob)        

        all_bootstrap_results.append({
          "model": model_key,
          "bootstrap_idx": b,
          "subgroup": group_id
        } | group_perf_metrics | group_cf_harm)

  return pd.DataFrame(all_bootstrap_results)