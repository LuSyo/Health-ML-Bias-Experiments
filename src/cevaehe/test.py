import torch
import numpy as np
import pandas as pd
from cevaehe.causal_validation import calculate_te_error, latent_recon_loss, run_sens_classifier, evaluate_latent_utility_fidelity, counterfactual_sensitivity
from metrics import calculate_performance_metrics, stratified_perf, get_cca

def test_ceveahe(model, test_loader, logger, args):

  device = args.device
  model.eval()

  all_y_true, all_y_pred_prob, all_y_full_cf_prob, all_sens, all_u_corr, all_u_desc, all_x_desc, all_x_desc_pred, all_x_desc_cf, all_x_corr, all_u_desc_cf = \
    [], [], [], [], [], [], [], [], [], [], []

  with torch.no_grad():
    for batch in test_loader:
      x_ind, x_desc, x_corr, x_sens, y = [t.to(device) for t in batch[:5]]

      s_bio = x_sens.clone()
      s_soc = x_sens.clone()

      # Infer latent variables
      mu_corr, _, mu_desc, _ = model.encode(x_desc, x_corr, x_ind, s_bio, s_soc, y=None)

      # Factual and Full Counterfactual prediction
      # from mean Ucorr and Udesc
      x_desc_pred_logits, _, y_pred_logits, x_desc_cf, _, _, _, y_full_cf_logits = model.decode(mu_desc, mu_corr, x_ind, s_bio, s_soc)

      x_desc_pred = model.hard_reconstruct_features(x_desc_pred_logits, model.desc_meta)
      
      # Soc. Counterfactual pass to adbuct the counterfactual Udesc
      # (for invariance test)
      s_soc_flipped = 1 - s_soc
      _, _, mu_desc_cf, _ = model.encode(x_desc_cf, x_corr, x_ind, s_bio, s_soc_flipped, y=None)

      all_y_true.append(y.cpu().numpy())
      all_y_pred_prob.append(torch.sigmoid(y_pred_logits).cpu().numpy())
      all_y_full_cf_prob.append(torch.sigmoid(y_full_cf_logits).cpu().numpy())
      all_sens.append(x_sens.cpu().numpy())
      all_u_corr.append(mu_corr.cpu().numpy())
      all_u_desc.append(mu_desc.cpu().numpy())
      all_x_desc.append(x_desc.cpu().numpy())
      all_x_desc_pred.append(x_desc_pred.cpu().numpy())
      all_x_desc_cf.append(x_desc_cf.cpu().numpy())
      all_x_corr.append(x_corr.cpu().numpy())
      all_u_desc_cf.append(mu_desc_cf.cpu().numpy())
  
  test_outputs = pd.DataFrame({
      'y_true': np.concatenate(all_y_true).flatten(),
      'y_pred_prob': np.concatenate(all_y_pred_prob).flatten(),
      'y_full_cf_prob': np.concatenate(all_y_full_cf_prob).flatten(),
      'sens': np.concatenate(all_sens).flatten(),
      'u_corr': list(np.concatenate(all_u_corr)),
      'u_desc': list(np.concatenate(all_u_desc)),
      'u_desc_cf': list(np.concatenate(all_u_desc_cf)),
      'x_desc': list(np.concatenate(all_x_desc)),
      'x_desc_pred': list(np.concatenate(all_x_desc_pred)),
      'x_desc_cf': list(np.concatenate(all_x_desc_cf)),
      'x_corr': list(np.concatenate(all_x_corr)),
  })
  test_outputs['y_pred'] = (test_outputs['y_pred_prob'] > 0.5).astype(int)

  perf_metrics = calculate_performance_metrics(
    test_outputs['y_true'],
    test_outputs['y_pred'],
    test_outputs['y_pred_prob'])
  
  strat_perf_metrics = stratified_perf(
    test_outputs['y_true'],
    test_outputs['y_pred'],
    test_outputs['y_pred_prob'],
    test_outputs['sens'],
  )

  ## Verify invariance of Udesc by Sens
  # 1. Sex classifier performance
  u_desc_sens_auc_mean, u_desc_sens_auc_std = run_sens_classifier(
    np.stack(test_outputs['u_desc']),
    test_outputs['sens'],
    args.seed
  )

  x_desc_sens_auc_mean, x_desc_sens_auc_std = run_sens_classifier(
    np.stack(test_outputs['x_desc']),
    test_outputs['sens'],
    args.seed
  )

  u_corr_sens_auc_mean, u_corr_sens_auc_std = run_sens_classifier(
    np.stack(test_outputs['u_corr']),
    test_outputs['sens'],
    args.seed
  )

  x_corr_sens_auc_mean, x_corr_sens_auc_std = run_sens_classifier(
    np.stack(test_outputs['x_corr']),
    test_outputs['sens'],
    args.seed
  )

  # 2. Counterfactual invariance
  u_desc_recon_loss = latent_recon_loss(
    np.stack(test_outputs['u_desc']), 
    np.stack(test_outputs['u_desc_cf']))

  # 3. Verify utility of Udesc
  desc_fidelity_scores = evaluate_latent_utility_fidelity(
    np.stack(test_outputs['u_desc']),
    np.stack(test_outputs['x_desc']),
    model.desc_meta,
    seed=args.seed
  )

  corr_fidelity_scores = evaluate_latent_utility_fidelity(
    np.stack(test_outputs['u_corr']),
    np.stack(test_outputs['x_corr']),
    model.corr_meta,
    seed=args.seed
  )

  logger.info("--- Latent Utility Fidelity ---")
  for feature, score in desc_fidelity_scores.items():
    logger.info(f"Desc: {feature} ({score['score_type']}): {score['score']:.4f}")
  for feature, score in corr_fidelity_scores.items():
    logger.info(f"Corr: {feature} ({score['score_type']}): {score['score']:.4f}")
  
  ## Calculate the Total Effect Error
  te_error, obs_disparity, est_ate, internal_te_error = calculate_te_error(
      test_outputs['y_true'].values,
      test_outputs['y_pred_prob'].values,
      test_outputs['y_full_cf_prob'].values,
      test_outputs['sens'].values
  )

  # Counterfactual sensitivity of Xdesc features
  sensitivity_results = counterfactual_sensitivity(
    np.stack(test_outputs['x_desc_pred']),
    np.stack(test_outputs['x_desc_cf']),
    model.desc_meta
  )

  logger.info("--- Xdesc Counterfactual Sensitivity ---")
  for feature, score in sensitivity_results.items():
    logger.info(f'{feature} score ({score['score_type']}): {score['score']:.4f}')
    perf_metrics[f"{feature}_ccs_{score['score_type']}"] = score['score']

  ####
  logger.info(f'Test Accuracy: {perf_metrics['accuracy']:.4f}')
  logger.info(f'Test AUC: {perf_metrics['roc_auc']:.4f}')
  logger.info(f'Test Brier Score: {perf_metrics['brier_score']:.4f}')
  logger.info(f'Test False Negative Rate: {perf_metrics['fnr']:.4f}')
  logger.info(f'Test False Positive Rate: {perf_metrics['fpr']:.4f}')
  logger.info(f'Udesc -> Xsens, ROC AUC: {u_desc_sens_auc_mean:.4f} (std. {u_desc_sens_auc_std:.4f})')
  logger.info(f'Xdesc -> Xsens, ROC AUC: {x_desc_sens_auc_mean:.4f} (std. {x_desc_sens_auc_std:.4f})')
  logger.info(f'Ucorr -> Xsens, ROC AUC: {u_corr_sens_auc_mean:.4f} (std. {u_corr_sens_auc_std:.4f})')
  logger.info(f'Xcorr -> Xsens, ROC AUC: {x_corr_sens_auc_mean:.4f} (std. {x_corr_sens_auc_std:.4f})')
  logger.info(f'Udesc counterfactual reconstruction loss: {u_desc_recon_loss:.4f}')
  logger.info(f'Observed Disparity (S=1 vs S=0): {obs_disparity:.4f}')
  logger.info(f'Estimated ATE: {est_ate:.4f}')
  logger.info(f'Total Effect (TE) Error: {te_error:.4f}')
  logger.info(f'Internal Total Effect (TE) Error: {internal_te_error:.4f}')

  # CAUSAL MODEL VALIDATION
  # Independence of u_desc and u_corr
  u_desc_matrix = np.stack(test_outputs['u_desc'].values)
  u_corr_matrix = np.stack(test_outputs['u_corr'].values)
  perf_metrics["u_u_cca"] = get_cca(u_desc_matrix, u_corr_matrix)
  logger.info(f'Canonical Correlation Analysis between U_corr and U_desc: {perf_metrics["u_u_cca"]:.4f}')

  perf_metrics['u_desc_sens_auc_mean'] = u_desc_sens_auc_mean
  perf_metrics['x_desc_sens_auc_mean'] = x_desc_sens_auc_mean
  perf_metrics['u_corr_sens_auc_mean'] = u_corr_sens_auc_mean
  perf_metrics['x_corr_sens_auc_mean'] = x_corr_sens_auc_mean
  perf_metrics['u_desc_recon_loss'] = u_desc_recon_loss
  perf_metrics['obs_disparity'] = obs_disparity
  perf_metrics['est_ate'] = est_ate
  perf_metrics['te_error'] = te_error
  perf_metrics['internal_te_error'] = internal_te_error

  return test_outputs, perf_metrics, strat_perf_metrics

def generate_fair_dataset(model, dataset, feature_mapping, args):
  """
    Generates a parallel CSV of latents and counterfactuals indexed to the original data.
  """
  model.eval()
  device = args.device

  m_samples = args.m_samples
  
  col_ind = [f['name'] for f in feature_mapping['ind']]
  col_desc = [f['name'] for f in feature_mapping['desc']]
  col_corr = [f['name'] for f in feature_mapping['corr']] 
  col_sens = [f['name'] for f in feature_mapping['sens']]
  target_name = feature_mapping['target']['name']

  all_counterfactuals = []
  all_latents = []
  batch_size = args.batch_size

  with torch.no_grad():
      for i in range(0, len(dataset), batch_size):
          j = min(i+batch_size, len(dataset))
          batch_df = dataset.iloc[i:j]
          
          # Convert to tensors
          x_ind = torch.tensor(batch_df[col_ind].values, dtype=torch.float32).to(device)
          x_desc = torch.tensor(batch_df[col_desc].values, dtype=torch.float32).to(device)
          x_corr = torch.tensor(batch_df[col_corr].values, dtype=torch.float32).to(device)
          s_bio = torch.tensor(batch_df[col_sens].values, dtype=torch.float32).to(device)
          s_soc = s_bio.clone().to(device)
          y = torch.tensor(batch_df[target_name].values, dtype=torch.float32).view(-1, 1).to(device)

          ## INFERENCE PASS 
          # To generate latent variable samples and counterfactual features
          # Using y=None to invoke inference encoders q(u|x, s)
          mu_c_inf, logvar_c_inf, mu_d_inf, logvar_d_inf = model.encode(x_desc, x_corr, x_ind, s_bio, s_soc, y=None)        
          
          _, _, _, x_desc_cf, x_corr_cf, *_ = model.decode(mu_d_inf, mu_c_inf, x_ind, s_bio, s_soc)

          u_c_samples = model.sample_latent(mu_c_inf, logvar_c_inf, m_samples)
          u_d_samples = model.sample_latent(mu_d_inf, logvar_d_inf, m_samples)  

          ## ABDUCTION PASS
          # To generate counterfactual outcome for IECO Ground Truth
          # Using y=y to invoke adbuction encoders q(u|x, s, y)
          _, _, mu_d_abd, _ = model.encode(x_desc, x_corr, x_ind, s_bio, s_soc, y=y)
          
          # Get Sociological Counterfactual Outcome Y'
          _, _, _, _, _, y_soc_cf_logits, _, _ = model.decode(mu_d_abd, mu_c_inf, x_ind, s_bio, s_soc)
          y_soc_cf_prob = torch.sigmoid(y_soc_cf_logits)

          # DATASETS CONSTRUCTION
          # Counterfactual variables and outcomes
          ref_index = batch_df.index.to_numpy()
          batch_cf = pd.DataFrame(index=ref_index)

          # CF Outcomes and Reconstructions
          batch_cf['y_soc_cf_prob'] = y_soc_cf_prob.cpu().numpy().flatten()

          # # x_desc_cf contains multiple features; map them back to names
          for i, feature in enumerate(feature_mapping['desc']):
              batch_cf[feature['name']] = x_desc_cf[:, i].cpu().numpy()
          for i, feature in enumerate(feature_mapping['corr']):
              batch_cf[feature['name']] = x_corr_cf[:, i].cpu().numpy()

          all_counterfactuals.append(batch_cf)

          # Latent variables 
          u_c_samples_df = process_latent_samples(u_c_samples, ref_index, 'u_c').reset_index(drop=True)   
          u_d_samples_df = process_latent_samples(u_d_samples, ref_index, 'u_d').reset_index(drop=True)   
          batch_latents = u_c_samples_df.merge(u_d_samples_df, left_index=True, right_index=True, suffixes=(None,'_dup')).drop(['patient_index_dup'], axis=1)

          all_latents.append(batch_latents)


  # Concatenate and reset index
  counterfactuals_df = pd.concat(all_counterfactuals)  
  latent_spaces_df = pd.concat(all_latents).reset_index(drop=True)

  return counterfactuals_df, latent_spaces_df

def process_latent_samples(samples, patient_indices, latent_name):
  '''
    Converts a list of M sample tensors into a long-format dataframe conserving the patient indices
  '''
  # samples_3d = np.array(samples)
  _, m_samples, latent_dim = samples.shape

  # Reshape to (batch_size, M_samples, latent_dim)
  # => keeps all samples for one patient together
  reshaped_samples = samples.reshape(-1, latent_dim)

  # Create repeated patient indices
  repeated_indices = np.repeat(patient_indices, m_samples)

  # Create dataframe
  column_names = [f"{latent_name}_dim{i}" for i in range(latent_dim)]
  df = pd.DataFrame(reshaped_samples, columns=column_names)
  df.insert(0, 'patient_index', repeated_indices)

  return df