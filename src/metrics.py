import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, brier_score_loss

def calculate_performance_metrics(y_true, y_pred, y_prob):
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred).flatten()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'brier_score': brier_score_loss(y_true, y_prob)
    }