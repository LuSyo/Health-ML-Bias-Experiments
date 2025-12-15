from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
import pandas as pd

def get_perf_metrics(y_true, y_pred, y_pred_proba):
  accuracy = accuracy_score(y_true, y_pred)
  roc_auc = roc_auc_score(y_true, y_pred_proba)
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  FNR = fn / (fn + tp)
  FPR = fp / (fp + tn)
  
  return [accuracy, roc_auc, FNR, FPR, tn, fp, fn, tp]


def get_audit_dataset(X_test, y_test, y_pred):
  dataset = X_test.copy()
  dataset['ground_truth'] = y_test
  dataset['prediction'] = y_pred

  return dataset

def evaluate_group(dataset, group_col, group_value):
  # Slice the target group
  group_dataset = dataset[dataset[group_col] == group_value]
  group_pred = group_dataset['prediction']
  group_truth = group_dataset['ground_truth']

  # Calculate performance metrics
  acc = accuracy_score(group_truth, group_pred)
  prec = precision_score(group_truth, group_pred, zero_division=0)
  rec = recall_score(group_truth, group_pred)

  # Confusion matrix
  tn, fp, fn, tp = confusion_matrix(group_truth, group_pred).ravel()

  print(f"--- Performance for: {group_col}={group_value} (N={len(group_dataset)}) ---")
  print(f"Accuracy:  {acc:.3f}")
  print(f"Recall:    {rec:.3f}")
  print(f"Precision: {prec:.3f}")
  print("\n")

  return {'accuracy':acc,
          'precision':prec,
          'recall':rec,
          'tn':tn,
          'fp':fp,
          'fn':fn,
          'tp':tp}

