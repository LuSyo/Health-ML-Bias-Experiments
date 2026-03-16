import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss
from sklearn.cross_decomposition import CCA

def calculate_performance_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        'brier_score': brier_score_loss(y_true, y_prob)
    }

def stratified_perf(y_true, y_pred, y_prob, sens):
    group_0 = sens == 0
    group_1 = sens == 1

    perf_metrics_0 = calculate_performance_metrics(
        y_true[group_0],
        y_pred[group_0],
        y_prob[group_0]
    )
    perf_metrics_1 = calculate_performance_metrics(
        y_true[group_1],
        y_pred[group_1],
        y_prob[group_1]
    )

    return {
        "accuracy_0": perf_metrics_0['accuracy'],
        "roc_auc_0": perf_metrics_0['roc_auc'],
        "fnr_0": perf_metrics_0['fnr'],
        "fpr_0": perf_metrics_0['fpr'],
        "brier_score_0": perf_metrics_0['brier_score'],
        "accuracy_1": perf_metrics_1['accuracy'],
        "roc_auc_1": perf_metrics_1['roc_auc'],
        "fnr_1": perf_metrics_1['fnr'],
        "fpr_1": perf_metrics_1['fpr'],
        "brier_score_1": perf_metrics_1['brier_score']
    }

def get_cca(set_a, set_b):
    cca = CCA(n_components=1)
    # projections of set_a and set_b projected in the 1D space 
    # where they are maximally correlated
    set_a_c, set_b_c = cca.fit_transform(set_a, set_b)

    # Return Pearson correlation coefficient between the projections across samples
    return np.corrcoef(set_a_c.T, set_b_c.T)[0, 1]

def get_interp_tpr(y_true, y_prob, mean_fpr):
    """
    Calculates TPR interpolated to a common FPR grid.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0 
    return interp_tpr