import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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