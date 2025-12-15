import semopy
import pandas as pd

def get_causal_model_params(X_train, y_train, y_pred_proba, protected_attribute):
  causal_features = pd.DataFrame()
  causal_features['protected_attribute'] = X_train[protected_attribute]
  causal_features['y_true'] = y_train
  causal_features['y_pred'] = y_pred_proba

  model_desc='''
    y_pred ~ y_true + protected_attribute
  '''

  causal_model = semopy.ModelMeans(model_desc)
  causal_model.fit(causal_features)
  causal_params = causal_model.inspect()

  # Retrieve the coefficients of the causal model
  beta2 = causal_params[causal_params.rval == "protected_attribute"]['Estimate'].values[0]
  beta2_pvalue = causal_params[causal_params.rval == "protected_attribute"]['p-value'].values[0]

  return [beta2, beta2_pvalue]