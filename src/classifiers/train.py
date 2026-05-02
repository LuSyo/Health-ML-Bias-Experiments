import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV

def initial_hyperparam_tuning(X, y, seed=4):
  param_grid = {
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True]
  }
  search = RandomizedSearchCV(
      estimator=RandomForestClassifier(
        n_estimators=500, 
        random_state=seed, 
        n_jobs=1
      ), 
      param_distributions=param_grid, 
      n_iter=15, 
      scoring='f1',
      cv=3, 
      n_jobs=1,
      random_state=seed
  )
  search.fit(X, y)
  return search.best_params_

def train_random_forest(X_train, y_train, X_test, X_cf_test, params, target_ppv, seed):
  '''
    Trains a sklearn RandomForestClassifier on the given training data,\
     optimised hyperparameters with 3-fold GridSearchCV

     Inputs:
       X_train: training features
       y_train: training labels
       X_test: test features
       y_test: test labels

     Outputs:
       rf: trained RandomForestClassifier
       y_pred: predicted labels
       y_pred_proba: predicted probabilities
  '''

  if params is None:
        params = {
            "max_depth": 10,
            "max_features": "sqrt",
            "min_samples_split": 5,
            "min_samples_leaf": 2
        }

  #create the RF classifier
  rf = RandomForestClassifier(
    **params, 
    n_estimators=500, 
    random_state=seed, 
    n_jobs=1
  )
  
  rf.fit(X_train, y_train)
  y_train_pred_proba = rf.predict_proba(X_train)[:, 1]
  threshold = find_threshold_at_target_ppv(y_train, y_train_pred_proba, target_ppv)

  y_pred_proba = rf.predict_proba(X_test)[:, 1]
  y_cf_pred_proba = rf.predict_proba(X_cf_test)[:, 1]

  y_pred = (y_pred_proba > threshold).astype(int)

  return [rf, y_pred, y_pred_proba, y_cf_pred_proba, threshold]

def find_threshold_at_target_ppv(y_true, y_probs, target_ppv):
  """
    finds the classification threshold to achieve the target PPV across the predictions
  """

  precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

  valid_indices = np.where(precisions >= target_ppv)[0]

  

  # If no valid indice is found, return the highest threshold
  if len(valid_indices) == 0:
    return thresholds[-1]

  chosen_idx = valid_indices[0]
  if chosen_idx < len(thresholds):
    return thresholds[chosen_idx]
  else:
    return thresholds[-1]
  