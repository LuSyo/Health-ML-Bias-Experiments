import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error

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
    sensitivity_results = []
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

      sensitivity_results.append({'name': f_name,
                                  'score': score,
                                  'score_type':score_type})

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

def evaluate_latent_utility_fidelity(u_desc, x_desc, desc_meta, seed=4):
  """
  Trains linear probes to predict each original feature in X_desc from U_desc.
  Quantifies how much clinical information survived the latent bottleneck.
  
  Inputs:
    u_desc: numpy array of the extracted latent variables
    x_desc: numpy array of the original descendant features
    desc_meta: list of dictionaries describing the features (from config/mapping)
    
  Outputs:
    fidelity_scores: Dictionary of scores for each feature
  """
  fidelity_scores = {}
  
  # Iterate through each feature in the X_desc bucket
  for i, feature_meta in enumerate(desc_meta):
      feature_name = feature_meta['name']
      feature_type = feature_meta['type']
      target_y = x_desc[:, i]
      
      if feature_type in ['categorical', 'binary']:
          # Classification Probe
          probe = LogisticRegression(max_iter=1000, random_state=seed)
          cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
          
          # Use ROC AUC for binary, or Accuracy for multi-class
          scoring = 'roc_auc' if feature_type == 'binary' else 'accuracy'
          
          scores = cross_val_score(probe, u_desc, target_y, cv=cv, scoring=scoring)
          fidelity_scores[f"{feature_name}_{scoring}"] = scores.mean()
          
      elif feature_type == 'continuous':
          # Regression Probe
          probe = Ridge(random_state=seed)
          cv = KFold(n_splits=5, shuffle=True, random_state=seed)
          
          # Use R-squared to measure variance explained
          scores = cross_val_score(probe, u_desc, target_y, cv=cv, scoring='r2')
          fidelity_scores[f"{feature_name}_r2"] = scores.mean()
          
  return fidelity_scores