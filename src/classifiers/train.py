from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def initial_hyperparam_tuning(X, y, seed=4):
  param_grid = {"max_depth": [5, 10, 20, None], "max_features": ["sqrt", "log2"]}
  search = RandomizedSearchCV(
      RandomForestClassifier(random_state=seed), 
      param_grid, n_iter=5, cv=3, n_jobs=-1
  )
  search.fit(X, y)
  return search.best_params_

def train_random_forest(X_train, y_train, X_test, params, seed):
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
    n_estimators=100, 
    random_state=seed, 
    n_jobs=2
  )
  
  rf.fit(X_train, y_train)
  y_pred = rf.predict(X_test)
  y_pred_proba = rf.predict_proba(X_test)[:, 1]

  return [y_pred, y_pred_proba]