from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_logreg(X_train, X_test, y_train, y_test):  
  steps = [('imputer',SimpleImputer()),
        ('scaler',StandardScaler()),
        ('classifier',LogisticRegression())]
  pipeline = Pipeline(steps)

  pipeline.fit(X_train, y_train)
  joblib.dump(pipeline, "models/baseline_logreg.joblib")

  y_pred = pipeline.predict(X_test)
  y_pred_probs = pipeline.predict_proba(X_test)

  # training and test performance
  test_accuracy = accuracy_score(y_test, y_pred)

  y_train_pred = pipeline.predict(X_train)
  train_accuracy = accuracy_score(y_train, y_train_pred)

  print(f"Model trained. Training accuracy: {train_accuracy}. Test accuracy: {test_accuracy}")
  
  return [pipeline, y_pred, y_pred_probs]

def train_random_forest(X_train, X_test, y_train, y_test):
  param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5]
  }
  
  #create the RF classifier
  rf = RandomForestClassifier()

  #create the grid search
  rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='roc_auc', cv=3)

  #fit the grid search
  rf_grid.fit(X_train, y_train)
  y_pred = rf_grid.predict(X_test)
  y_pred_proba = rf_grid.predict_proba(X_test)[:,1]

  return [rf_grid, y_pred, y_pred_proba]

