import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss, average_precision_score
from sklearn.cross_decomposition import CCA

def calculate_performance_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        'brier_score': brier_score_loss(y_true, y_prob)
    }

def stratified_perf(y_true, y_pred, y_pred_prob, sens, y_cf_prob=None, y_cf_pred_prob=None):
    group_0 = sens == 0
    group_1 = sens == 1

    perf_metrics_0 = calculate_performance_metrics(
        y_true[group_0],
        y_pred[group_0],
        y_pred_prob[group_0]
    )
    perf_metrics_1 = calculate_performance_metrics(
        y_true[group_1],
        y_pred[group_1],
        y_pred_prob[group_1]
    )

    strat_metrics = {  
        "accuracy_0": perf_metrics_0['accuracy'],
        "roc_auc_0": perf_metrics_0['roc_auc'],
        "auprc_0": perf_metrics_0['auprc'],
        "fnr_0": perf_metrics_0['fnr'],
        "fpr_0": perf_metrics_0['fpr'],
        "brier_score_0": perf_metrics_0['brier_score'],
        "accuracy_1": perf_metrics_1['accuracy'],
        "roc_auc_1": perf_metrics_1['roc_auc'],
        "auprc_1": perf_metrics_1['auprc'],
        "fnr_1": perf_metrics_1['fnr'],
        "fpr_1": perf_metrics_1['fpr'],
        "brier_score_1": perf_metrics_1['brier_score']
    }

    if y_cf_prob is not None and y_cf_pred_prob is not None:
        ieco_mace_0, _ = calculate_ieco_mace(
            y_true[group_0],
            y_cf_prob[group_0],
            y_pred_prob[group_0],
            y_cf_pred_prob[group_0]
        )
        ieco_mace_1, _ = calculate_ieco_mace(
            y_true[group_1],
            y_cf_prob[group_1],
            y_pred_prob[group_1],
            y_cf_pred_prob[group_1]
        )
        strat_metrics['ieco_mace_0'] = ieco_mace_0
        strat_metrics['ieco_mace_1'] = ieco_mace_1


    return strat_metrics

def avg_perf_per_patient(y_true, y_pred_prob, y_cf_prob, y_cf_pred_prob, sens, patient_index):
    test_results = pd.DataFrame({
        'patient_index': patient_index,
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'y_cf_prob': y_cf_prob,
        'y_cf_pred_prob': y_cf_pred_prob,
        'sens': sens
      })

    avg_results: pd.DataFrame = test_results.groupby('patient_index').mean()

    y_pred_avg = (avg_results['y_pred_prob'] >= 0.5).astype(int)

    global_perf = calculate_performance_metrics(
        avg_results['y_true'],
        y_pred_avg,
        avg_results['y_pred_prob']
    )

    global_ieco_mace, _ = calculate_ieco_mace(
        avg_results['y_true'],
        avg_results['y_cf_prob'],
        avg_results['y_pred_prob'],
        avg_results['y_cf_pred_prob']
    )
    global_perf['ieco_mace'] = global_ieco_mace

    strat_perf = stratified_perf(
        avg_results['y_true'],
        y_pred_avg,
        avg_results['y_pred_prob'],
        avg_results['sens'],
        avg_results['y_cf_prob'],
        avg_results['y_cf_pred_prob']
    )

    grouped_roc_curve = get_grouped_roc_curve(
        avg_results['y_true'],
        avg_results['y_pred_prob'],
        avg_results['sens'],
    )

    return global_perf, strat_perf, grouped_roc_curve

def get_cca(set_a, set_b):
    if set_a is None or set_b is None or set_a.shape[1] == 0 or set_b.shape[1] == 0:
        return np.nan
    
    if len(set_a.shape) == 1:
        set_a = set_a.reshape(-1, 1)
    if len(set_b.shape) == 1:
        set_b = set_b.reshape(-1, 1)
    
    cca = CCA(n_components=1)
    # projections of set_a and set_b projected in the 1D space 
    # where they are maximally correlated
    set_a_c, set_b_c = cca.fit_transform(set_a, set_b)

    # Return Pearson correlation coefficient between the projections across samples
    return np.corrcoef(set_a_c.T, set_b_c.T)[0, 1]

def get_interp_tpr(y_true, y_prob):
    """
    Calculates TPR interpolated to a common FPR grid.
    """
    mean_fpr = np.linspace(0, 1, 100)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0 
    return interp_tpr

def get_grouped_roc_curve(y_true, y_prob, sens):
    grouped_roc_curve = {}
    for group in np.unique(sens):
        mask = (sens == group)
        grouped_roc_curve[group] = get_interp_tpr(y_true[mask], y_prob[mask])
    
    return grouped_roc_curve

def calculate_ieco_mace(y_true, y_cf_prob, y_pred_prob, y_pred_cf_prob):
  '''
    Measures the Equalised Counterfactual Odds criteria for counterfactual fairness via Mean Absolute Counterfactual Error:
    An individual whose sensitive attribute is changed in a counterfactual world, all independent factors being the same, should receive the same prediction as their image, given that they have the same actual outcome. 

    Inputs:
      - y_true: The factual actual outcome (as binary)
      - y_cf: The counterfactual actual outcome (as probabilities)
      - y_pred: The factual prediction (as probabilities)
      - y_pred_cf: The counterfactual prediction (as probabilities)

    Outputs:
      mace: Mean Absolute Counterfactual Error
  '''
  total_mace = np.mean(np.abs(y_pred_prob - y_pred_cf_prob))

  # IECO MACE, conditioned on equality of factual and counterfactual outcome
  y_cf = (y_cf_prob > 0.5).astype(int)
  equal_outcome = y_cf == y_true

  if not equal_outcome.any():
    ieco_mace = np.nan
  else:
    ieco_mace = np.mean(np.abs(y_pred_prob - y_pred_cf_prob)[equal_outcome])

  return ieco_mace, total_mace
    