from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_random_forest(X_train, y_train, X_test):
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
  param_grid = {
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
  }

  #create the RF classifier
  rf = RandomForestClassifier(random_state=4, n_estimators=100)

  #create the grid search
  rf_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                               n_iter=10, scoring='roc_auc',
                               cv=3, n_jobs=-1, random_state=4)

  #fit the grid search
  rf_search.fit(X_train, y_train)
  y_pred = rf_search.predict(X_test)
  y_pred_proba = rf_search.predict_proba(X_test)[:,1]

  return [rf_search, y_pred, y_pred_proba]