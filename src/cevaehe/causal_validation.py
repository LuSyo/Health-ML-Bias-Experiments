import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score
import torch
from torch import nn

from cevaehe.model import CEVAEHE
from cevaehe.data_loader import make_bucketed_loader
from cevaehe.train import lite_train_ceveahe
from metrics import calculate_ieco_mace

def run_sens_classifier(features, target_sens, seed=4):
  '''
    Trains a Random Forest classifier on the given features\
     to predict the target sensitive attribute.

    Inputs
      features: Pandas DataFrame of features
      target_sex: target sensitive attribute to predict

    Outputs
      roc_auc: ROC AUC (Receiver Operating Characteristic Area Under the Curve)
  '''
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  audit_rf = RandomForestClassifier(
      n_estimators=300,
      min_samples_leaf=5,
      max_features=features.shape[1],
      random_state=seed
  )
  scores = cross_val_score(audit_rf, features, target_sens, cv=cv, scoring='roc_auc')
  return scores.mean(), scores.std()

def latent_recon_loss(u, u_cf):
  '''
    Calculates the mean reconstruction loss between the latent U and its counterfactual, as a Mean Square Error loss

    Inputs
      u: the factual latent vector
      u_cf: the counterfactual latent vector

    Outputs
      recon_loss: the mean reconstruction loss across dimensions of the latent
  '''
  recon_loss = 0
  dim = u.shape[1]
  for i in range(dim):
    recon_loss += mean_squared_error(u[:,i], u_cf[:,i] )

  return recon_loss / dim

def counterfactual_sensitivity(v_pred, v_pred_cf, v_meta):
    '''
      Calculates Counterfactual Sensitivity scores for the given feature bucket, matching each feature type to the correct scoring
    '''
    sample_size = v_pred.shape[0]
    sensitivity_results = {}
    for i, feature in enumerate(v_meta):
      f_name = feature['name']
      f_type = feature['type']

      v = v_pred[:, i]
      v_cf = v_pred_cf[:, i]

      if f_type in ['categorical', 'binary']:
        # Flip Rate: Percentage of cases where the hard-reconstructed 
        # category or bit changed
        changes = np.sum(v != v_cf)
        score = changes / sample_size
        score_type = "flip rate"
      else:
        # Continuous: Mean Absolute Deviation (MAD)
        score = np.mean(np.abs(v - v_cf))
        score_type = "MAD"

      sensitivity_results[f_name] = {
                                  'score': score,
                                  'score_type':score_type
                                  }

    return sensitivity_results

def calculate_te_error(y_true, y_pred_prob, y_cf_prob, sens):
  '''
    Calculates the Total Effect (TE) Error between observed group outcome disparities \
    and the total effect estimated by the model's counterfactual generation
  '''

  # Factual disparity in outcomes
  # E[Y|S=1] - E[Y|S=0]
  group_0_mask = sens == 0
  group_1_mask = sens == 1

  obs_disparity = y_true[group_1_mask].mean() - y_true[group_0_mask].mean()

  # Estimated Average Total Effect (ATE)
  # E[Y(do(S=1)) - Y(do(S=0))]
  # If factual sens == 1, Y(do(S=1)) = factual prediction, Y(do(S=0)) = counterfactual prediction  
  # If factual sens == 0, Y(do(S=0)) = factual prediction, Y(do(S=1)) = counterfactual prediction
  y_do_1 = np.where(group_1_mask, y_pred_prob, y_cf_prob)
  y_do_0 = np.where(group_0_mask, y_pred_prob, y_cf_prob)

  est_ate = (y_do_1 - y_do_0).mean()

  # TE Error
  te_error = abs(obs_disparity - est_ate)

  # Internal consistency check
  # i.e. TE error against the model's reconstructed outcome
  model_disparity = y_pred_prob[group_1_mask].mean() - y_pred_prob[group_0_mask].mean()
  internal_te_error = abs(model_disparity - est_ate)

  return te_error, obs_disparity, est_ate, internal_te_error

def evaluate_latent_utility_fidelity(u, x, x_meta, seed=4):
  """
  Trains linear probes to predict each original feature in X_desc from U_desc.
  Quantifies how much clinical information survived the latent bottleneck.
  
  Inputs:
    u: numpy array of the extracted latent variables
    x: numpy array of the original descendant features
    x_meta: list of dictionaries describing the features (from config/mapping)
    
  Outputs:
    fidelity_scores: Dictionary of scores for each feature
  """
  fidelity_scores = {}
  
  # Iterate through each feature in the X_desc bucket
  for i, feature_meta in enumerate(x_meta):      
      feature_name = feature_meta['name']
      feature_type = feature_meta['type']
      target_y = x[:, i]

      cv_folds = 5
      scoring = ""
      scores = np.zeros(cv_folds)
      norm_score = 0
      
      if feature_type in ['categorical', 'binary']:
        # Classification Probe
        probe = LogisticRegression(max_iter=1000, random_state=seed)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        if feature_type == 'binary':
          scoring = 'roc_auc'
          scores = cross_val_score(probe, u, target_y, cv=cv, scoring=scoring)
          avg_score = scores.mean()
          norm_score = (avg_score - 0.5) / 0.5
        else:
          counts = np.bincount(target_y.astype(int))
          base_acc = np.max(counts) / len(target_y)
        
          scoring = 'accuracy'
          scores = cross_val_score(probe, u, target_y, cv=cv, scoring=scoring)
          norm_score = (scores.mean() - base_acc) / (1.0 - base_acc) if base_acc < 1.0 else 0.0
          
      elif feature_type == 'continuous':
        # Regression Probe
        probe = Ridge(random_state=seed)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        
        # Use R-squared to measure variance explained
        scoring = 'r2'
        scores = cross_val_score(probe, u, target_y, cv=cv, scoring=scoring)
        norm_score = scores.mean()
          
      fidelity_scores[feature_name] = {
         'score': scores.mean(),
         'score_type': scoring,
         'norm_score': max(0, norm_score)
      }

  return fidelity_scores


def run_sps_bootstrap(dataset, feature_mapping, iterations, lite_epochs, logger, args):
  """
  Stochastic pathway sensitivity (SPS) audit
  Iteratively re-partition features into Xdesc and Xcorr to test causal fidelity
  """

  results = []

  logger.info("--- Start Stochastic bucket sensitivity (SBS) audit ---")

  features = feature_mapping['X']

  # establish baseline with all features in Xcorr
  baseline_mapping = {
    'desc': [],
    'corr': features,
    'ind': feature_mapping['ind'],
    'sens': feature_mapping['sens'],
    'target': feature_mapping['target']
  }

  train_loader, val_loader, _ \
      = make_bucketed_loader(dataset, baseline_mapping, val_size=0.3,
                              test_size=0, batch_size=args.batch_size, seed=args.seed)
    
  model = CEVAEHE(
    ind_meta=baseline_mapping['ind'], 
    desc_meta=baseline_mapping['desc'], 
    corr_meta=baseline_mapping['corr'], 
    sens_meta=baseline_mapping['sens'], 
    args=args
  ).to(args.device)

  lite_train_ceveahe(
    model,
    train_loader,
    lite_epochs,
    logger,
    args
  )

  baseline_results = sps_test_ceveahe(model, val_loader, features, [], -1,
                                         logger, args)
  
  for i in range(iterations):
    logger.info(f"> Iteration {i+1}/{iterations}")
    
    # 1. Randomly partition features
    # We ensure at least one feature remains in each bucket
    shuffled = random.sample(features, len(features))
    split_point = random.randint(1, len(shuffled) - 1)
    
    current_desc = shuffled[:split_point]
    current_corr = shuffled[split_point:]

    # Construct the current mapping
    current_mapping = {
      'desc': current_desc,
      'corr': current_corr,
      'ind': feature_mapping['ind'],
      'sens': feature_mapping['sens'],
      'target': feature_mapping['target']
    }

    # Data loaders
    train_loader, val_loader, _ \
      = make_bucketed_loader(dataset, current_mapping, val_size=0.3,
                              test_size=0, batch_size=args.batch_size, seed=args.seed)
    
    # 2. Initialize Model with current SCM
    model = CEVAEHE(
        ind_meta=current_mapping['ind'], 
        desc_meta=current_mapping['desc'], 
        corr_meta=current_mapping['corr'], 
        sens_meta=current_mapping['sens'], 
        args=args
    ).to(args.device)

    logger.info(f'U_corr dimension: {model.uc_dim}')
    logger.info(f'U_desc dimension: {model.ud_dim}')

    # 3. Lite train model
    lite_train_ceveahe(
        model,
        train_loader,
        lite_epochs,
        logger,
        args
    )

    # 4. Measure metrics for this causal model
    iteration_results = sps_test_ceveahe(model, val_loader, features, current_desc, i,
                                         logger, args)

    results.extend(iteration_results)
          
  return pd.DataFrame(baseline_results), pd.DataFrame(results)

def sps_test_ceveahe(model, test_loader, features, desc_features, iteration, logger, args):
  model.eval()
  all_sens, all_u_corr, all_u_desc, all_x_corr, all_x_desc, all_x_desc_pred, all_x_desc_cf, all_y_true, all_y_cf_prob, all_y_full_cf_prob, all_y_pred_prob, all_y_pred_cf_prob = \
  [], [], [], [], [], [], [], [], [], [], [], []

  is_baseline = len(desc_features) == 0
  if is_baseline: logger.info(f'---BASELINE RESULTS---')

  with torch.no_grad():
    for batch in test_loader:
      x_ind, x_desc, x_corr, x_sens, y = [t.to(args.device) for t in batch[:5]]

      s_bio = x_sens.clone()
      s_soc = x_sens.clone()

      #--- ABDUCTION ---
      # Get the latent variables, the reconstructed features, 
      # the counterfactual outcomes
      mu_corr, _, mu_desc, _ = model.encode(x_desc, x_corr, x_ind, s_bio, s_soc, y)

      # Factual and Full Counterfactual prediction
      # from mean Ucorr and Udesc
      x_desc_pred_logits, _, _, x_desc_cf, x_corr_cf, y_soc_cf_logits, y_bio_cf_logits, y_full_cf_logits = model.decode(mu_desc, mu_corr, x_ind, s_bio, s_soc)

      x_desc_pred = model.hard_reconstruct_features(x_desc_pred_logits, model.desc_meta)

      y_cf_logits = y_bio_cf_logits if is_baseline else y_soc_cf_logits
      y_cf_prob = nn.Sigmoid()(y_cf_logits)

      y_full_cf_prob = nn.Sigmoid()(y_full_cf_logits)

      # --- FACTUAL INFERENCE ---
      # Get the factual prediction
      mu_corr_inf, _, mu_desc_inf, _ = model.encode(x_desc, x_corr, x_ind, s_bio, s_soc, y=None)
      _, _, y_pred_inf_logits, *_= model.decode(mu_desc_inf, mu_corr_inf, x_ind, s_bio, s_soc)

      y_pred_prob = nn.Sigmoid()(y_pred_inf_logits)

      # --- SYNTHETIC COUNTERFACTUAL INFERENCE ---
      # Get the counterfactual prediction
      x_desc_syn = x_desc if is_baseline else x_desc_cf
      x_corr_syn = x_corr_cf if is_baseline else x_corr
      s_bio_cf = 1 - s_bio if is_baseline else s_bio
      s_soc_cf = s_soc if is_baseline else 1 - s_soc

      mu_corr_cf_inf, _, mu_desc_cf_inf, _ = model.encode(x_desc_syn, x_corr_syn, x_ind, s_bio_cf, s_soc_cf, y=None)
      _, _, y_pred_cf_inf_logits, *_= model.decode(mu_desc_cf_inf, mu_corr_cf_inf, x_ind, s_bio_cf, s_soc_cf)

      y_pred_cf_prob = nn.Sigmoid()(y_pred_cf_inf_logits)

      # --- STORE BATCH RESULTS ---
      all_sens.append(x_sens.cpu().numpy())
      all_u_corr.append(mu_corr.cpu().numpy())
      all_u_desc.append(mu_desc.cpu().numpy())
      all_x_desc.append(x_desc.cpu().numpy())
      all_x_corr.append(x_corr.cpu().numpy())
      all_y_true.append(y.cpu().numpy())
      all_y_cf_prob.append(y_cf_prob.cpu().numpy())
      all_y_full_cf_prob.append(y_full_cf_prob.cpu().numpy())
      all_y_pred_prob.append(y_pred_prob.cpu().numpy())
      all_y_pred_cf_prob.append(y_pred_cf_prob.cpu().numpy())

      batch_size = x_corr.shape[0]
      if x_desc_pred is not None:
        all_x_desc_pred.append(x_desc_pred.cpu().numpy())
      else:
        all_x_desc_pred.append(np.empty((batch_size, 0)))
          
      if x_desc_cf is not None:
        all_x_desc_cf.append(x_desc_cf.cpu().numpy())
      else:
        all_x_desc_cf.append(np.empty((batch_size, 0)))
    
  sens_np = np.concatenate(all_sens).flatten()
  u_corr_np = np.stack(list(np.concatenate(all_u_corr)))
  u_desc_np = np.stack(list(np.concatenate(all_u_desc)))
  x_desc_np = np.stack(list(np.concatenate(all_x_desc)))
  x_corr_np = np.stack(list(np.concatenate(all_x_corr)))
  x_desc_pred_np = np.stack(list(np.concatenate(all_x_desc_pred)))
  x_desc_cf_np = np.stack(list(np.concatenate(all_x_desc_cf)))
  y_true_np = np.concatenate(all_y_true).flatten()
  y_cf_prob_np = np.concatenate(all_y_cf_prob).flatten()
  y_full_cf_prob_np = np.concatenate(all_y_full_cf_prob).flatten()
  y_pred_prob_np = np.concatenate(all_y_pred_prob).flatten()
  y_pred_cf_prob_np = np.concatenate(all_y_pred_cf_prob).flatten()

  # Latent Utility Fidelity
  f_desc = evaluate_latent_utility_fidelity(
      u_desc_np, x_desc_np, model.desc_meta, args.seed
  )

  f_corr = evaluate_latent_utility_fidelity(
      u_corr_np, x_corr_np, model.corr_meta, args.seed
  )

  # Counterfactual sensitivity
  cf_sensitivity = counterfactual_sensitivity(
      x_desc_pred_np, x_desc_cf_np, model.desc_meta
  )

  # # Total Effect Error
  te_error, *_ = calculate_te_error(
    y_true_np,
    y_pred_prob_np,
    y_full_cf_prob_np,
    sens_np
  )

  # IECO MACE
  stratified_ieco_mace = {}
  stratified_total_mace = {}
  for v in np.unique(sens_np):
    group_mask = sens_np == v
    group_ieco_mace, group_total_mace = calculate_ieco_mace(
      y_true_np[group_mask],
      y_cf_prob_np[group_mask], 
      y_pred_prob_np[group_mask], 
      y_pred_cf_prob_np[group_mask]
    )
    stratified_ieco_mace["ieco_mace_" + str(v)] = group_ieco_mace
    stratified_total_mace["total_mace_" + str(v)] = group_total_mace
    logger.info(f'TOTAL MACE, GROUP {str(v)}: {group_total_mace}')

  ieco_mace, total_mace = calculate_ieco_mace(y_true_np, y_cf_prob_np, y_pred_prob_np, y_pred_cf_prob_np)
  logger.info(f'TOTAL MACE: {ieco_mace}')

  # Utility: AUPRC
  roc_auc = roc_auc_score(y_true_np, y_pred_prob_np)
  auprc = average_precision_score(y_true_np, y_pred_prob_np)
  logger.info(f'ROC AUC: {roc_auc}')
  logger.info(f'AUPRC: {auprc}')

  results = []
  for feature in features:
    f_name = feature['name']
    
    f_desc_score = f_desc[f_name]['score'] if f_desc.get(f_name, False) else np.nan
    f_corr_score = f_corr[f_name]['score'] if f_corr.get(f_name, False) else np.nan
    norm_f_desc_score = f_desc[f_name]['norm_score'] if f_desc.get(f_name, False) else np.nan
    norm_f_corr_score = f_corr[f_name]['norm_score'] if f_corr.get(f_name, False) else np.nan
    sensitivity = cf_sensitivity[f_name]['score'] if cf_sensitivity.get(f_name, False) else np.nan

    results.append({
      "iteration": iteration,
      "feature": f_name,
      "bucket": "x_desc" if feature in desc_features else "x_corr",
      "roc_auc": roc_auc,
      "auprc": auprc,
      "te_error": te_error,
      "ieco_mace": ieco_mace,
      "total_mace": total_mace,
      "cf_sensitivity": sensitivity,
      "sensitivity_scoring": cf_sensitivity[f_name]['score_type'] if cf_sensitivity.get(f_name, False) else "",
      "f_desc": f_desc_score,
      "f_corr": f_corr_score,
      "norm_f_desc": norm_f_desc_score,
      "norm_f_corr": norm_f_corr_score,
      "fidelity_scoring": f_desc[f_name]['score_type'] if f_desc.get(f_name, False) else f_corr[f_name]['score_type'],
      "desc_size": len(desc_features),
      "u_desc_dim": model.ud_dim
    } | stratified_ieco_mace | stratified_total_mace)
    
  return results