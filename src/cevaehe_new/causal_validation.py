import random
import gc
from typing import Tuple, cast, List, Any

import torch
from torch import nn

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from fairlearn.metrics import equalized_odds_difference

from cevaehe_new.model import CEVAEHE
from cevaehe_new.data_loader import make_bucketed_loader
from cevaehe_new.train import train_cevaehe
from metrics import calculate_counterfactual_harm, get_baseline_bce
from utils import set_global_seeds

def run_downstream_probe(features, target, sens, dict_prefix="", cf_features=None, seed=4):
  """
  Trains a probe once per fold and evaluates it across factual global, subgroups, and counterfactual test distributions

  Inputs:
      features: 2D np.ndarray of factual features
      target: 1D np.ndarray of true outcomes
      sens_attr: 1D np.ndarray containing sensitive attribute
      cf_features: 2D np.ndarray of counterfactual features (optional)
  """
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  sens_flat = sens.flatten()
  subgroups = np.unique(sens_flat).astype(int)
  threshold = target.sum()/len(target)

  global_auprc = []
  global_recall = []
  global_precision = []
  global_eod = []
  global_f1 = []
  global_cf_metrics = {"bal_harm": [], "harm_pos": [], "harm_neg": []}
  subgroup_auprc = {g: [] for g in subgroups}
  subgroup_recall = {g: [] for g in subgroups}
  subgroup_precision = {g: [] for g in subgroups}
  subgroup_f1 = {g: [] for g in subgroups}
  subgroup_cf_metrics = {
      g: {"bal_harm": [], "harm_pos": [], "harm_neg": []} 
      for g in subgroups
  }

  for train_idx, test_idx in cv.split(features, target):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = target[train_idx], target[test_idx]
    s_test = sens_flat[test_idx]
    
    # Fit probe exactly once per fold
    probe = RandomForestClassifier(
      n_estimators=300,
      min_samples_leaf=5,
      random_state=seed
    )
    probe.fit(X_train, y_train)
    
    # Factual Global Score
    y_pred_prob = probe.predict_proba(X_test)[:, 1] #type: ignore
    y_pred = (y_pred_prob > threshold).astype(int)
    global_auprc.append(average_precision_score(y_test, y_pred_prob))
    global_recall.append(recall_score(y_test, y_pred))
    global_precision.append(precision_score(y_test, y_pred))
    global_f1.append(f1_score(y_test, y_pred))
    global_eod.append(equalized_odds_difference(
      y_test, y_pred,
      sensitive_features=s_test,
      method="between_groups",
      agg="worst_case"))

    # Global Counterfactual Harm Evaluation
    if cf_features is not None:
      X_test_cf = cf_features[test_idx]
      y_pred_prob_cf = probe.predict_proba(X_test_cf)[:, 1] #type: ignore
      
      global_cf_harm = calculate_counterfactual_harm(
        y_test,
        y_pred_prob,
        y_pred_prob_cf
      )
      global_cf_metrics["bal_harm"].append(global_cf_harm['cf_harm_balanced'])
      global_cf_metrics["harm_pos"].append(global_cf_harm['cf_harm_pos'])
      global_cf_metrics["harm_neg"].append(global_cf_harm['cf_harm_neg'])
    
    # Stratified evaluation
    for g in subgroups:
      subgroup_mask = (s_test == g)

      if np.sum(subgroup_mask) == 0:
        continue

      y_test_sub = y_test[subgroup_mask]
      y_pred_prob_sub = y_pred_prob[subgroup_mask] 
      y_pred_sub = y_pred[subgroup_mask] 

      if y_test_sub.sum() > 0:
        subgroup_auprc[g].append(average_precision_score(y_test_sub, y_pred_prob_sub))  
        subgroup_recall[g].append(recall_score(y_test_sub, y_pred_sub))
      else:
        subgroup_auprc[g].append(np.nan)
        subgroup_recall[g].append(np.nan)

      subgroup_precision[g].append(precision_score(y_test_sub, y_pred_sub))   
      subgroup_f1[g].append(f1_score(y_test_sub, y_pred_sub))   

      if cf_features is not None:
        y_pred_prob_cf_sub = y_pred_prob_cf[subgroup_mask] #type: ignore
        
        sub_cf_harm = calculate_counterfactual_harm(
            y_test_sub,
            y_pred_prob_sub,
            y_pred_prob_cf_sub
        )
        subgroup_cf_metrics[g]["bal_harm"].append(sub_cf_harm['cf_harm_balanced'])
        subgroup_cf_metrics[g]["harm_pos"].append(sub_cf_harm['cf_harm_pos'])
        subgroup_cf_metrics[g]["harm_neg"].append(sub_cf_harm['cf_harm_neg'])   
    

  # AGGREGATED METRICS
  results = {
    f"{dict_prefix}global_mean_auprc": np.nanmean(global_auprc),
    f"{dict_prefix}global_std_auprc": np.nanstd(global_auprc),
    f"{dict_prefix}global_mean_recall": np.nanmean(global_recall),
    f"{dict_prefix}global_std_recall": np.nanstd(global_recall),
    f"{dict_prefix}global_mean_precision": np.nanmean(global_precision),
    f"{dict_prefix}global_std_precision": np.nanstd(global_precision),
    f"{dict_prefix}global_mean_f1": np.nanmean(global_f1),
    f"{dict_prefix}global_std_f1": np.nanstd(global_f1),
    f"{dict_prefix}global_mean_eod": np.nanmean(global_eod),
    f"{dict_prefix}global_std_eod": np.nanstd(global_eod),
  }
  
  if cf_features is not None:
    results[f"{dict_prefix}bal_harm_mean"] = np.nanmean(global_cf_metrics['bal_harm'])
    results[f"{dict_prefix}bal_harm_std"] = np.nanstd(global_cf_metrics['bal_harm'])
    results[f"{dict_prefix}harm_pos_mean"] = np.nanmean(global_cf_metrics['harm_pos'])
    results[f"{dict_prefix}harm_pos_std"] = np.nanstd(global_cf_metrics['harm_pos'])
    results[f"{dict_prefix}harm_neg_mean"] = np.nanmean(global_cf_metrics['harm_neg'])
    results[f"{dict_prefix}harm_neg_std"] = np.nanstd(global_cf_metrics['harm_neg'])

  for g in subgroups:
    results[f'{dict_prefix}{g}_mean_auprc'] = np.nanmean(subgroup_auprc[g])
    results[f'{dict_prefix}{g}_std_auprc'] = np.nanstd(subgroup_auprc[g])
    results[f'{dict_prefix}{g}_mean_recall'] = np.nanmean(subgroup_recall[g])
    results[f'{dict_prefix}{g}_std_recall'] = np.nanstd(subgroup_recall[g])
    results[f'{dict_prefix}{g}_mean_precision'] = np.nanmean(subgroup_precision[g])
    results[f'{dict_prefix}{g}_std_precision'] = np.nanstd(subgroup_precision[g])
    results[f'{dict_prefix}{g}_mean_f1'] = np.nanmean(subgroup_f1[g])
    results[f'{dict_prefix}{g}_std_f1'] = np.nanstd(subgroup_f1[g])

    if cf_features is not None:
      results[f"{dict_prefix}{g}_bal_harm_mean"] = np.nanmean(subgroup_cf_metrics[g]['bal_harm'])
      results[f"{dict_prefix}{g}_bal_harm_std"] = np.nanstd(subgroup_cf_metrics[g]['bal_harm'])
      results[f"{dict_prefix}{g}_harm_pos_mean"] = np.nanmean(subgroup_cf_metrics[g]['harm_pos'])
      results[f"{dict_prefix}{g}_harm_pos_std"] = np.nanstd(subgroup_cf_metrics[g]['harm_pos'])
      results[f"{dict_prefix}{g}_harm_neg_mean"] = np.nanmean(subgroup_cf_metrics[g]['harm_neg'])
      results[f"{dict_prefix}{g}_harm_neg_std"] = np.nanstd(subgroup_cf_metrics[g]['harm_neg'])
      
  return results

def run_test_classifier(features, target, scoring="average_precision",seed=4):
  '''
    Trains a Random Forest classifier on the given features\
     to predict the target.

    Inputs
      features: Pandas DataFrame of features
      target: target to predict
      scoring: scoring function used by the classifier (default = "average_precision")

    Outputs
      scoring metric mean and std
  '''
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  audit_rf = RandomForestClassifier(
      n_estimators=300,
      min_samples_leaf=2,
      class_weight="balanced_subsample",
      random_state=seed
  )
  scores = cross_val_score(audit_rf, features, target, cv=cv, scoring=scoring)

  return scores.mean(), scores.std()

def _get_config_key(desc_list):
  """Generates a unique, order-independent key for a pathway configuration."""
  return tuple(sorted([f['name'] if isinstance(f, dict) else f for f in desc_list]))

def _audit_pathway_config(dataset, mapping, iteration, run_id, all_features, logger, args):
  # 1. Data loaders
  train_loader, val_loader = make_bucketed_loader(
    dataset, 
    mapping, 
    val_size=0.2, 
    batch_size=args.batch_size, 
    seed=args.seed)
  
  # 2. Initialize Model with current SCM
  model = CEVAEHE(
    desc_meta=mapping['desc'], 
    sens_meta=mapping['sens'], 
    args=args
  ).to(args.device)

  logger.info(f'U_desc dimension: {model.ud_dim}')

  # 3. Lite train model
  train_cevaehe(
    model,
    train_loader,
    val_loader,
    logger,
    args
  )

  set_global_seeds(args.seed)

  full_loader, _ = make_bucketed_loader(
    dataset, 
    mapping, 
    val_size=0, 
    batch_size=args.batch_size, 
    seed=args.seed)

  results = sps_test_ceveahe(
    model, 
    full_loader, 
    all_features, 
    mapping['desc'], 
    iteration, 
    run_id, 
    logger, 
    args
  )
  
  del model
  del train_loader
  del val_loader
  
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  
  gc.collect()
  
  return results

def run_sps_bootstrap(dataset, feature_mapping, logger, args):
  """
  Stochastic pathway sensitivity (SPS) audit
  Iteratively re-partition features into Xdesc and Xcorr to test causal fidelity
  """
  n_cross_val = args.cross_val if args.cross_val > 1 else 1

  results = []
  baseline_results = []

  logger.info("--- Start Stochastic bucket sensitivity (SBS) audit ---")

  features = feature_mapping['X']

  targeted_pathways = feature_mapping.get('targeted_pathways', [])
  seen_keys = set()

  # establish baseline with all features in Xdesc
  baseline_mapping = {
    'desc': features,
    'indcorr': feature_mapping['ind'],
    'sens': feature_mapping['sens'],
    'target': feature_mapping['target']
  }
  baseline_key = _get_config_key([])
  seen_keys.add(baseline_key)

  for j in range(n_cross_val):
    logger.info(f'BASELINE CONFIG')
    logger.info(f"Bootstrap #{j+1}/{n_cross_val}")
    boot_dataset = dataset.sample(frac=1.0, axis=0, random_state=args.seed + j)
    baseline_run_results = _audit_pathway_config(boot_dataset, baseline_mapping, -1, j,features, logger, args)
    baseline_results.extend(baseline_run_results)

  # TARGETED PATHWAYS
  for i, tp in enumerate(targeted_pathways):
    config_name = tp.get('name', f"targeted_{i}")

    current_mapping = {
      'desc': tp["desc"],
      'indcorr': tp["indcorr"],
      'sens': feature_mapping['sens'],
      'target': feature_mapping['target']
    }

    for j in range(n_cross_val):
      logger.info(f"Running targeted pathway: {config_name}")
      logger.info(f"Bootstrap #{j+1}/{n_cross_val}")
      boot_dataset = dataset.sample(frac=1.0, axis=0, random_state=args.seed + j)
      tp_results = _audit_pathway_config(boot_dataset, current_mapping, config_name, j, features, logger, args)
      results.extend(tp_results)

    seen_keys.add(_get_config_key(tp['desc']))

  # Pre-generate unique configurations
  random_configs = []
  attempts = 0
  max_attempts = args.sps_iter * 100

  while len(random_configs) < args.sps_iter and attempts < max_attempts:
    attempts += 1
    
    # Shuffle and partition
    shuffled = random.sample(features, len(features))
    split_point = random.randint(1, len(shuffled)- 1)
    
    current_desc = shuffled[:split_point]
    current_corr = shuffled[split_point:]
    
    current_key = _get_config_key(current_desc)
    
    # Only keep it if it's genuinely new (and doesn't match baseline/targeted keys)
    if current_key not in seen_keys:
      seen_keys.add(current_key)
      random_configs.append((current_desc, current_corr))

  if len(random_configs) < args.sps_iter:
    logger.warning(
        f"Sampling limit hit."
        f"Generated {len(random_configs)} unique configs instead of the requested {args.sps_iter}."
    )
  
  for i, (current_desc, current_corr) in enumerate(random_configs):

    # Construct the current mapping
    current_mapping = {
      'desc': current_desc,
      'indcorr': current_corr + feature_mapping['ind'],
      'sens': feature_mapping['sens'],
      'target': feature_mapping['target']
    }

    for j in range(n_cross_val):
      logger.info(f'Iteration #{i+1}/{len(random_configs)}')
      logger.info(f"Bootstrap #{j+1}/{n_cross_val}")
      boot_dataset = dataset.sample(frac=1.0, axis=0, random_state=args.seed + j)
      iteration_results = _audit_pathway_config(boot_dataset, current_mapping, i, j,features, logger, args)
      results.extend(iteration_results)
          
  return pd.DataFrame(baseline_results), pd.DataFrame(results)

def sps_test_ceveahe(model, test_loader, features, desc_features, iteration, run_id, logger, args):
  device = args.device
  model.eval()

  all_y_true, all_x_sens, all_x_indcorr, all_u_desc, all_x_desc, all_x_desc_cf, all_y_pred_cf, all_pred_logits =\
    [], [], [], [], [], [], [], []

  all_x_desc_recon_logits = {}

  # RANDOM-LEVEL Y_RECON_L
  test_y_np = test_loader.dataset.tensors[3].cpu().numpy()
  y_random_bce, y_prevalence = get_baseline_bce(test_y_np)
  test_s_np = test_loader.dataset.tensors[2].cpu().numpy()
  s_prevalence = np.mean(test_s_np)

  with torch.no_grad():
    for batch in test_loader:
      x_indcorr, x_desc, x_sens, y = [t.to(device) for t in batch[:4]]

      mu_desc, logvar_desc = model.encode(x_desc, x_sens, y=None)

      if mu_desc.isnan().any():
        logger.error("NaNs detected in latent.")
        raise Exception("CEVAEHE collapse. Check scaling of training and test datasets.")

      u_desc = model.reparameterize(mu_desc, logvar_desc)

      x_desc_recon_logits, y_pred_logits, x_desc_recon_cf, y_pred_cf_logits = model.decode(
        u_desc, x_sens
      )

      all_y_true.append(y)
      all_x_sens.append(x_sens)
      all_u_desc.append(u_desc)
      all_x_desc.append(x_desc)
      all_x_indcorr.append(x_indcorr)
      all_x_desc_cf.append(x_desc_recon_cf)
      all_y_pred_cf.append(torch.sigmoid(y_pred_cf_logits))
      all_pred_logits.append(y_pred_logits)
    
  all_y_true = torch.cat(all_y_true, dim=0)
  all_x_desc = torch.cat(all_x_desc, dim=0)
  all_x_desc_cf = torch.cat(all_x_desc_cf, dim=0)
  all_x_indcorr = torch.cat(all_x_indcorr, dim=0)
  all_u_desc = torch.cat(all_u_desc, dim=0)
  all_x_sens = torch.cat(all_x_sens, dim=0)
  all_pred_logits = torch.cat(all_pred_logits, dim=0)

  all_y_true_np = all_y_true.cpu().numpy().flatten()
  all_x_desc_np = all_x_desc.cpu().numpy()
  all_x_desc_cf_np = all_x_desc_cf.cpu().numpy()
  all_x_indcorr_np = all_x_indcorr.cpu().numpy()
  all_u_desc_np = all_u_desc.cpu().numpy()
  all_x_sens_np = all_x_sens.cpu().numpy().reshape(-1, 1)
  
  if len(desc_features):
    
    # Probe trained on (X_indcorr, X_desc)
    # Global AUPRC 
    x_probe_input = np.concatenate([all_x_indcorr_np, all_x_desc_np], axis=1)
    x_probe_cf_input = np.concatenate([all_x_indcorr_np, all_x_desc_cf_np], axis=1)

    x_results = run_downstream_probe(
      features = x_probe_input,
      target = all_y_true_np,
      sens = all_x_sens_np,
      cf_features = x_probe_cf_input,
      dict_prefix="x_",
      seed = args.seed
    )
    
    # Probe trained on (X_indcorr, U_desc)
    # Global AUPRC 
    u_probe_input = np.concatenate([all_x_indcorr_np, all_u_desc_np], axis=1)

    u_results = run_downstream_probe(
      features = u_probe_input,
      target = all_y_true_np,
      sens = all_x_sens_np,
      dict_prefix="u_",
      seed = args.seed
    )

    # Probe trained on (X_indcorr) = ablation
    # Global AUPRC 
    abla_results = run_downstream_probe(
      features = all_x_indcorr_np,
      target = all_y_true_np,
      sens = all_x_sens_np,
      dict_prefix="abla_",
      seed = args.seed
    )

    # S probes
    mean_x_s_bal_acc, std_x_s_bal_acc = run_test_classifier(
      features=all_x_desc_np,
      target=all_x_sens_np.ravel(),
      scoring="balanced_accuracy",
      seed=args.seed
    )

    mean_u_s_bal_acc, std_u_s_bal_acc = run_test_classifier(
      features=all_u_desc_np,
      target=all_x_sens_np.ravel(),
      scoring="balanced_accuracy",
      seed=args.seed
    )

    sens_probe_results = {
      "mean_x_s_bal_acc": mean_x_s_bal_acc,
      "std_x_s_bal_acc": std_x_s_bal_acc,
      "mean_u_s_bal_acc": mean_u_s_bal_acc,
      "std_u_s_bal_acc": std_u_s_bal_acc,
    }

  else:
    u_results = {}
    x_results = {}
    abla_results = {}
    sens_probe_results = {}

  results = []
  for feature in features:
    f_name = feature['name']

    results.append({
      "iteration": iteration,
      "run_id": run_id,
      "feature": f_name,
      "bucket": "x_desc" if feature in desc_features else "x_corr",
      "y_prevalence": y_prevalence,
      "s_prevalence": s_prevalence
    } | u_results | x_results | abla_results | sens_probe_results)

  return results

