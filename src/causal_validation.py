from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error

def run_sens_classifier(features, target_sens):
  '''
    Trains a Random Forest classifier on the given features\
     to predict the target sensitive attribute.

    Inputs
      features: Pandas DataFrame of features
      target_sex: target sensitive attribute to predict

    Outputs
      roc_auc: ROC AUC (Receiver Operating Characteristic Area Under the Curve)
  '''
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
  audit_rf = RandomForestClassifier(
      n_estimators=100,
      max_depth=5,
      random_state=4
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