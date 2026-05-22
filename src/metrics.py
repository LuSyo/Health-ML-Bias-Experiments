import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, brier_score_loss, average_precision_score
from sklearn.cross_decomposition import CCA

def calculate_performance_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        'brier_score': brier_score_loss(y_true, y_prob),
        'recall': tp / (fn + tp) if (fn + tp) > 0 else np.nan,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
    }

def stratified_perf(y_true, y_pred, y_pred_prob, sens, y_cf_pred_prob=None):
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
        "recall_0": perf_metrics_0['recall'],
        "ppv_0": perf_metrics_0['ppv'],
        "accuracy_1": perf_metrics_1['accuracy'],
        "roc_auc_1": perf_metrics_1['roc_auc'],
        "auprc_1": perf_metrics_1['auprc'],
        "fnr_1": perf_metrics_1['fnr'],
        "fpr_1": perf_metrics_1['fpr'],
        "brier_score_1": perf_metrics_1['brier_score'],
        "recall_1": perf_metrics_1['recall'],
        "ppv_1": perf_metrics_1['ppv'],
    }

    # if y_cf_pred_prob is not None:
    #     strat_metrics['cf_harm_0'], strat_metrics['cf_harm_pos_0'], strat_metrics['cf_harm_neg_0'] = calculate_counterfactual_harm(
    #         y_true[group_0],
    #         y_pred_prob[group_0],
    #         y_cf_pred_prob[group_0]
    #     )
    #     strat_metrics['cf_harm_1'], strat_metrics['cf_harm_pos_1'], strat_metrics['cf_harm_neg_1'] = calculate_counterfactual_harm(
    #         y_true[group_1],
    #         y_pred_prob[group_1],
    #         y_cf_pred_prob[group_1]
    #     )

    return strat_metrics

def avg_perf_per_patient(y_true, y_pred_prob, y_cf_pred_prob, sens, patient_index, threshold):
    test_results = pd.DataFrame({
        'patient_index': patient_index,
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'y_cf_pred_prob': y_cf_pred_prob,
        'sens': sens
      })

    avg_results: pd.DataFrame = test_results.groupby('patient_index').mean()

    y_pred_avg = (avg_results['y_pred_prob'] >= threshold).astype(int)

    global_perf = calculate_performance_metrics(
        avg_results['y_true'],
        y_pred_avg,
        avg_results['y_pred_prob']
    )

    global_perf['threshold'] = threshold

    # global_perf['cf_harm'], global_perf['cf_harm_pos'], global_perf['cf_harm_neg'] = calculate_counterfactual_harm(
    #     avg_results['y_true'],
    #     avg_results['y_pred_prob'],
    #     avg_results['y_cf_pred_prob']
    # )

    strat_perf = stratified_perf(
        avg_results['y_true'],
        y_pred_avg,
        avg_results['y_pred_prob'],
        avg_results['sens'],
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

def calculate_ieco_mace(y_true, y_cf_prob, y_pred_prob, y_pred_cf_prob, threshold=0.5):
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
    total_mace = calculate_mace(y_pred_prob, y_pred_cf_prob)

    # IECO MACE, conditioned on equality of factual and counterfactual outcome
    y_cf = (y_cf_prob > threshold).astype(int)
    equal_outcome = y_cf == y_true

    if not equal_outcome.any():
        ieco_mace = np.nan
    else:
        ieco_mace = np.mean(calculate_mace(y_pred_prob, y_pred_cf_prob)[equal_outcome])

    return ieco_mace, total_mace

def calculate_mace(y_pred_prob, y_cf_prob):
    return np.mean(np.abs(y_pred_prob - y_cf_prob))

def calculate_balanced_total_mace(y_true, y_pred_prob, y_cf_prob):
    """
    Calculates macro-averaged Total MACE to ensure imbalanced outcomes
    do not mask the causal signal in the minority class.
    """
    # Calculate absolute probability shifts for the whole set
    abs_diffs = np.abs(y_pred_prob - y_cf_prob)

    # Class-specific means
    mace_pos = np.mean(abs_diffs[y_true == 1])
    mace_neg = np.mean(abs_diffs[y_true == 0])

    # Macro-average (unweighted by class size)
    balanced_mace = (mace_pos + mace_neg) / 2

    return balanced_mace
    
def calculate_counterfactual_harm(y_true, y_pred_prob, y_pred_cf_prob):
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    # Harm to the sick: Risk is LOWER in counterfactual (Under-diagnosis)
    harm_pos = np.mean(np.maximum(0, y_pred_prob[pos_mask] - y_pred_cf_prob[pos_mask]))

    # Harm to the healthy: Risk is HIGHER in counterfactual (Over-diagnosis)
    harm_neg = np.mean(np.maximum(0, y_pred_cf_prob[neg_mask] - y_pred_prob[neg_mask]))

    # Balanced Harm metric
    balanced_harm = (harm_pos + harm_neg) / 2

    return balanced_harm, harm_pos, harm_neg

def get_baseline_bce(target_np):
    prevalence = np.mean(target_np)

    if 0 < prevalence < 1:
        baseline_bce = - (prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence))
    else:
        baseline_bce = 0.6931

    return baseline_bce, prevalence