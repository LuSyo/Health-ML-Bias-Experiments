import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import torch

from cevaehe.model import CEVAEHE
from cevaehe.data_loader import make_bucketed_loader
from cevaehe.train import lite_train_ceveahe

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
      n_estimators=100,
      max_depth=5,
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
      
      if feature_type in ['categorical', 'binary']:
        # Classification Probe
        probe = LogisticRegression(max_iter=1000, random_state=seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

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
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        
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

def run_sbs_bootstrap(dataset, feature_mapping, iterations, lite_epochs, logger, args):
  """
  Stochastic bucket sensitivity (SBS) audit
  Iteratively re-partition features into Xdesc and Xcorr to test causal fidelity
  """

  results = []

  logger.info("--- Start Stochastic bucket sensitivity (SBS) audit ---")

  for i in range(iterations):
    logger.info(f"> Iteration {i+1}/{iterations}")
    
    # 1. Randomly partition features
    # We ensure at least one feature remains in each bucket
    features = feature_mapping['X']
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
        ind_meta=feature_mapping['ind'], 
        desc_meta=current_desc, 
        corr_meta=current_corr, 
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
    model.eval()
    all_u_corr, all_u_desc, all_x_corr, all_x_desc, all_x_desc_pred, all_x_desc_cf = \
    [], [], [], [], [], []

    with torch.no_grad():
      for batch in val_loader:
        x_ind, x_desc, x_corr, x_sens, y = [t.to(args.device) for t in batch[:5]]

        s_bio = x_sens.clone()
        s_soc = x_sens.clone()

        # Abduct latent variables
        mu_corr, _, mu_desc, _ = model.encode(x_desc, x_corr, x_ind, s_bio, s_soc, y)

        # Factual and Full Counterfactual prediction
        # from mean Ucorr and Udesc
        x_desc_pred_logits, _, _, x_desc_cf, _, _, _, _ = model.decode(mu_desc, mu_corr, x_ind, s_bio, s_soc)

        x_desc_pred = model.hard_reconstruct_features(x_desc_pred_logits, model.desc_meta)

        all_u_corr.append(mu_corr.cpu().numpy())
        all_u_desc.append(mu_desc.cpu().numpy())
        all_x_desc.append(x_desc.cpu().numpy())
        all_x_corr.append(x_corr.cpu().numpy())
        all_x_desc_pred.append(x_desc_pred.cpu().numpy())
        all_x_desc_cf.append(x_desc_cf.cpu().numpy())
      
    u_corr_np = np.stack(list(np.concatenate(all_u_corr)))
    u_desc_np = np.stack(list(np.concatenate(all_u_desc)))
    x_desc_np = np.stack(list(np.concatenate(all_x_desc)))
    x_corr_np = np.stack(list(np.concatenate(all_x_corr)))
    x_desc_pred_np = np.stack(list(np.concatenate(all_x_desc_pred)))
    x_desc_cf_np = np.stack(list(np.concatenate(all_x_desc_cf)))

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

    for feature in features:
      f_name = feature['name']
      
      f_desc_score = f_desc[f_name]['score'] if f_desc.get(f_name, False) else np.nan
      f_corr_score = f_corr[f_name]['score'] if f_corr.get(f_name, False) else np.nan
      norm_f_desc_score = f_desc[f_name]['norm_score'] if f_desc.get(f_name, False) else np.nan
      norm_f_corr_score = f_corr[f_name]['norm_score'] if f_corr.get(f_name, False) else np.nan
      sensitivity = cf_sensitivity[f_name]['score'] if cf_sensitivity.get(f_name, False) else np.nan


      results.append({
        "iteration": i,
        "feature": f_name,
        "bucket": "x_desc" if feature in current_desc else "x_corr",
        "cf_sensitivity": sensitivity,
        "sensitivity_scoring": cf_sensitivity[f_name]['score_type'] if cf_sensitivity.get(f_name, False) else "",
        "f_desc": f_desc_score,
        "f_corr": f_corr_score,
        "norm_f_desc": norm_f_desc_score,
        "norm_f_corr": norm_f_corr_score,
        "fidelity_scoring": f_desc[f_name]['score_type'] if f_desc.get(f_name, False) else f_corr[f_name]['score_type'],
        "desc_size": len(current_desc),
        "u_desc_dim": model.ud_dim
      })
          
  return pd.DataFrame(results)