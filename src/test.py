from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch
import numpy as np
import pandas as pd
from src.metrics import calculate_performance_metrics

def test_dcevae(model, test_loader, logger, args):
  device = args.device
  model.eval()

  all_y_true, all_y_pred_prob, all_y_cf_prob, all_sens = [], [], [], []

  with torch.no_grad():
    for batch in test_loader:
      x_ind, x_desc, x_corr, x_sens, y = [t.to(device) for t in batch[:5]]

      # Infer latent variables
      mu_corr, _, mu_desc, _ = model.encode(x_ind, x_desc, x_corr, x_sens, y=None)

      # Factual and Counterfactual prediction
      # using the mean u_corr and u_desc
      u_corr = mu_corr
      u_desc = mu_desc

      _, _, y_pred_prob, _, y_cf_prob = model.decode(u_desc, u_corr, x_ind, x_sens)

      all_y_true.append(y.cpu().numpy())
      all_y_pred_prob.append(torch.sigmoid(y_pred_prob).cpu().numpy())
      all_y_cf_prob.append(torch.sigmoid(y_cf_prob).cpu().numpy())
      all_sens.append(x_sens.cpu().numpy())

  test_results = pd.DataFrame({
      'y_true': np.concatenate(all_y_true).flatten(),
      'y_pred_prob': np.concatenate(all_y_pred_prob).flatten(),
      'y_cf_prob': np.concatenate(all_y_cf_prob).flatten(),
      'sens': np.concatenate(all_sens).flatten()
  })
  test_results['y_pred'] = (test_results['y_pred_prob'] > 0.5).astype(int)

  perf_metrics = calculate_performance_metrics(
    test_results['y_true'],
    test_results['y_pred'],
    test_results['y_pred_prob'])

  logger.info(f'Test Accuracy: {perf_metrics['accuracy']:.4f}')
  logger.info(f'Test AUC: {perf_metrics['roc_auc']:.4f}')
  logger.info(f'Test False Negative Rate: {perf_metrics['fnr']:.4f}')
  logger.info(f'Test False Positive Rate: {perf_metrics['fpr']:.4f}')

  return test_results
