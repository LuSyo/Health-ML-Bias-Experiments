import torch
import numpy as np
import pandas as pd
from src.causal_validation import run_sens_classifier
from src.metrics import calculate_performance_metrics, stratified_perf

def test_dcevae(model, test_loader, logger, args):

  device = args.device
  model.eval()

  all_y_true, all_y_pred_prob, all_y_cf_prob, all_sens, all_u_corr, all_u_desc, all_x_desc = \
    [], [], [], [], [], [], []

  with torch.no_grad():
    for batch in test_loader:
      x_ind, x_desc, x_corr, x_sens, y = [t.to(device) for t in batch[:5]]

      # Infer latent variables
      mu_corr, _, mu_desc, _ = model.encode(x_desc, x_corr, x_ind, x_sens, y=None)

      # Infer counterfactual latent variables
      # _, _, mu_desc_cf, _ = model.encode(x_desc, x_corr, x_ind, x_sens, y=None)

      # Factual and Counterfactual prediction
      # from mean Ucorr and Udesc
      _, _, y_pred_prob, _, y_cf_prob = model.decode(mu_desc, mu_corr, x_ind, x_sens)

      all_y_true.append(y.cpu().numpy())
      all_y_pred_prob.append(torch.sigmoid(y_pred_prob).cpu().numpy())
      all_y_cf_prob.append(torch.sigmoid(y_cf_prob).cpu().numpy())
      all_sens.append(x_sens.cpu().numpy())
      all_u_corr.append(mu_corr.cpu().numpy())
      all_u_desc.append(mu_desc.cpu().numpy())
      all_x_desc.append(x_desc.cpu().numpy())
  
  test_results = pd.DataFrame({
      'y_true': np.concatenate(all_y_true).flatten(),
      'y_pred_prob': np.concatenate(all_y_pred_prob).flatten(),
      'y_cf_prob': np.concatenate(all_y_cf_prob).flatten(),
      'sens': np.concatenate(all_sens).flatten(),
      'u_corr': list(np.concatenate(all_u_corr)),
      'u_desc': list(np.concatenate(all_u_desc)),
      'x_desc': list(np.concatenate(all_x_desc)),
  })
  test_results['y_pred'] = (test_results['y_pred_prob'] > 0.5).astype(int)

  perf_metrics = calculate_performance_metrics(
    test_results['y_true'],
    test_results['y_pred'],
    test_results['y_pred_prob'])
  
  strat_perf_metrics = stratified_perf(
    test_results['y_true'],
    test_results['y_pred'],
    test_results['y_pred_prob'],
    test_results['sens'],
  )

  # Verify invariance of Ucorr by Sens
  # 1. Sex classifier performance
  u_desc_sens_auc_mean, u_desc_sens_auc_std = run_sens_classifier(
    np.stack(test_results['u_desc']),
    test_results['sens']
  )

  x_desc_sens_auc_mean, x_desc_sens_auc_std = run_sens_classifier(
    np.stack(test_results['x_desc']),
    test_results['sens']
  )

  # 2. Counterfactual invariance


  logger.info(f'Test Accuracy: {perf_metrics['accuracy']:.4f}')
  logger.info(f'Test AUC: {perf_metrics['roc_auc']:.4f}')
  logger.info(f'Test Brier Score: {perf_metrics['brier_score']:.4f}')
  logger.info(f'Test False Negative Rate: {perf_metrics['fnr']:.4f}')
  logger.info(f'Test False Positive Rate: {perf_metrics['fpr']:.4f}')
  logger.info(f'Udesc -> Xsens, ROC AUC: {u_desc_sens_auc_mean:4f} (std. {u_desc_sens_auc_std:4f})')
  logger.info(f'Xdesc -> Xsens, ROC AUC: {x_desc_sens_auc_mean:4f} (std. {x_desc_sens_auc_std:4f})')

  return test_results, perf_metrics, strat_perf_metrics
