from ucimlrepo import fetch_ucirepo
import pandas as pd
from tableone import TableOne

def prepare_kaggleuciheartdisease():
  heart_disease = pd.read_csv('data/heart_disease_uci.csv')
  X = heart_disease.drop('num', axis=1)
  y = heart_disease['num']

  # Generate Table 1
  table1 = generate_tableone(heart_disease, 
    groupby="sex", 
    continuous=["age","trestbps","chol","oldpeak" ],
    categorical=["cp","fbs","restecg","exang","slope","thal","num"])
  print(table1)

  # snake case categorical data values
  heart_disease = heart_disease\
    .map(lambda v: v.lower().replace(" ","_").replace("-","_") if isinstance(v,str) else v, na_action='ignore')

  # One-hot encoding for all categorical columns missing values
  encoded_na_columns = pd.get_dummies(X[['fbs','restecg','exang','slope','thal']], ['fbs','restecg','exang','slope','thal'], dtype=int)
  
  # One-hot encoding for all categorical columns without missing value
  encoded_columns = pd.get_dummies(heart_disease[['sex','dataset','cp']], ['sex','dataset','cp'], dtype=int, drop_first=True)

  heart_disease = pd.concat([heart_disease, encoded_na_columns, encoded_columns], axis=1)
  heart_disease.drop(['fbs','restecg','exang','slope','thal','sex','dataset','cp' ], axis=1, inplace=True)
  heart_disease = heart_disease.rename(columns={'sex_male':'sex'})

  # Drop ca
  heart_disease.drop('ca',axis=1,inplace=True)

  # Treat label column as a binary class, with any value > 0 set to 1
  heart_disease['num'] = heart_disease['num'].map(lambda v: 1 if v > 0 else 0)

  X = heart_disease.drop('num',axis=1)
  y = heart_disease['num'].to_numpy()

  # Create the stratas to analyse the dataset and model performance
  # Stratify by outcome and sex (protected attribute)
  stratas = y.astype(str) + "_" + X['sex'].astype(str)

  return [X, y, stratas]


def prepare_fetcheduciheartdisease():
  # fetch dataset
  heart_disease = fetch_ucirepo(id=45)

  # data (as pandas dataframes)
  X = heart_disease.data.features.copy()
  y = heart_disease.data.targets.copy()
    
  # Generate table 1
  full_dataset = X.copy()
  full_dataset['heartdisease'] = y.iloc[:,0].copy()
  table1 = generate_tableone(full_dataset, 
    groupby="sex", 
    group_values={0:'Female', 1:'Male'},
    continuous=["age","trestbps","chol","thalach","oldpeak" ],
    categorical=["cp","fbs","restecg","exang","slope","thal","heartdisease"])
  print(table1)

  # One-hot encoding for 'thal'
  X = X.replace({'thal':{3:"normal",6:"fixed",7:"reversible"}})
  thal_columns = pd.get_dummies(X['thal'], 'thal', dtype=int)

  X = pd.concat([X,thal_columns],axis=1)

  X.drop('thal', axis=1, inplace=True)

  # Imputation for 'ca'
  ca_mode = X['ca'].mode()
  X['ca'] = X['ca'].fillna(ca_mode[0])

  # Treat target label as a binary class
  y.loc[y['num'] > 0,'num'] = 1

  y = y.iloc[:,0].to_numpy()

  # Create the stratas to analyse the dataset and model performance
  # Stratify by outcome and sex (protected attribute)
  stratas = y.astype(str) + "_" + X['sex'].astype(str)

  return [X, y, stratas]

def generate_tableone(dataset, groupby=None, group_values=None, continuous=[], categorical=[]):
  if groupby and group_values:
    dataset[groupby] = dataset[groupby].replace(group_values)

  return TableOne(dataset, groupby=groupby, missing=False, continuous=continuous, categorical=categorical)