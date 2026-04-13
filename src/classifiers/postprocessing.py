import numpy as np
from sklearn.metrics import confusion_matrix

def fit_fair_thresholds(y_true_val, y_prob_val, sens_val, max_fnr_diff=0.02, steps=100):
    """
    Learns subgroup-specific classification thresholds to achieve FNR parity 
    (Equality of Opportunity) while maximizing overall utility (accuracy).
    
    Note: This MUST be fit on a validation set, not the test set, to avoid data leakage.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    
    group_0_mask = (sens_val == 0)
    group_1_mask = (sens_val == 1)
    
    def evaluate_thresholds(y_t, y_p):
        results = []
        for t in thresholds:
            y_pred = (y_p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            acc = (tp + tn) / len(y_t)
            results.append({'threshold': t, 'fnr': fnr, 'acc': acc})
        return results

    metrics_0 = evaluate_thresholds(y_true_val[group_0_mask], y_prob_val[group_0_mask])
    metrics_1 = evaluate_thresholds(y_true_val[group_1_mask], y_prob_val[group_1_mask])
    
    best_tau_0, best_tau_1 = 0.5, 0.5
    max_combined_acc = -1
    
    weight_0 = sum(group_0_mask) / len(sens_val)
    weight_1 = sum(group_1_mask) / len(sens_val)

    # Grid search for the optimal combination of thresholds
    for m0 in metrics_0:
        for m1 in metrics_1:
            fnr_diff = abs(m0['fnr'] - m1['fnr'])
            
            # Constraint: FNR disparity must be within the acceptable tolerance
            if fnr_diff <= max_fnr_diff:
                # Objective: Maximize population-weighted accuracy
                combined_acc = (m0['acc'] * weight_0) + (m1['acc'] * weight_1)
                
                if combined_acc > max_combined_acc:
                    max_combined_acc = combined_acc
                    best_tau_0 = m0['threshold']
                    best_tau_1 = m1['threshold']
                    
    if max_combined_acc == -1:
        print(f"Warning: Could not find thresholds satisfying max_fnr_diff <= {max_fnr_diff}. Defaulting to 0.5")
        return 0.5, 0.5

    return best_tau_0, best_tau_1

def predict_with_fair_thresholds(y_prob, sens, tau_0, tau_1):
    """
    Applies the learned subgroup-specific thresholds to generate hard classifications.
    """
    y_pred = np.zeros_like(y_prob, dtype=int)
    
    group_0_mask = (sens == 0)
    group_1_mask = (sens == 1)
    
    y_pred[group_0_mask] = (y_prob[group_0_mask] >= tau_0).astype(int)
    y_pred[group_1_mask] = (y_prob[group_1_mask] >= tau_1).astype(int)
    
    return y_pred