import torch
import torch.utils.data as utils
from sklearn.model_selection import train_test_split
import numpy as np

def make_loader(X_ind, X_desc, X_corr, X_sens, Y, index, batch_size=32):
  X_ind_fact = X_ind[index]
  X_desc_fact = X_desc[index]
  X_corr_fact = X_corr[index]
  X_sens_fact = X_sens[index]
  Y_fact = Y[index]

  # # Permuted set for the discriminator
  permuted_indices = np.random.permutation(X_ind_fact.shape[0])
  X_ind_perm = X_ind_fact[permuted_indices]
  X_desc_perm = X_desc_fact[permuted_indices]
  X_corr_perm = X_corr_fact[permuted_indices]
  X_sens_perm = X_sens_fact[permuted_indices]
  Y_perm = Y_fact[permuted_indices]

  X_ind_tensor = torch.tensor(X_ind_fact, dtype=torch.float32)
  X_desc_tensor = torch.tensor(X_desc_fact, dtype=torch.float32)
  X_corr_tensor = torch.tensor(X_corr_fact, dtype=torch.float32)
  X_sens_tensor = torch.tensor(X_sens_fact, dtype=torch.float32)
  Y_tensor = torch.tensor(Y_fact, dtype=torch.float32)
  X_ind_tensor_2 = torch.tensor(X_ind_perm, dtype=torch.float32)
  X_desc_tensor_2 = torch.tensor(X_desc_perm, dtype=torch.float32)
  X_corr_tensor_2 = torch.tensor(X_corr_perm, dtype=torch.float32)
  X_sens_tensor_2 = torch.tensor(X_sens_perm, dtype=torch.float32)
  Y_tensor_2 = torch.tensor(Y_perm, dtype=torch.float32)

  # dataset = utils.TensorDataset(X_ind_tensor, X_desc_tensor, X_corr_tensor, X_sens_tensor, Y_tensor)
  dataset = utils.TensorDataset(X_ind_tensor, X_desc_tensor, X_corr_tensor, X_sens_tensor, Y_tensor, X_ind_tensor_2, X_desc_tensor_2, X_corr_tensor_2, X_sens_tensor_2, Y_tensor_2)
  loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return loader

def make_bucketed_loader(dataset, map, val_size=0.2, test_size=0.1, batch_size=32, seed=4):
  '''
    Creates train, validation and test DataLoader for the given dataset, \
    separating features into independent, sensitive, descendant, and correlated features, and stratified by target class.

    Input:
      - dataset: a pandas DataFrame
      - map: a dictionary mapping feature names to buckets
      - val_size: the proportion of the dataset to use for validation
      - test_size: the proportion of the dataset to use for testing
      - batch_size: the batch size for the DataLoader
      - seed: a seed for the random number generator

    Output:
      - train_loader: Training DataLoader
      - val_loader: Validation DataLoader
      - test_loader: Testing DataLoader
      - ind_types: a list of the data types of the independent features
      - desc_types: a list of the data types of the descendant features
      - corr_types: a list of the data types of the correlated features
      - sens_type: the data type of the sensitive feature
  '''
  np.random.seed(seed=seed)

  ## BUCKET DATASET
  # Independent, Descendant, Correlated features and Sensitive attributes
  col_ind = []
  for feature in map['ind']:
    col_ind.append(feature['name'])
  X_ind = dataset[col_ind].to_numpy()

  col_desc = []
  for feature in map['desc']:
    col_desc.append(feature['name'])
  X_desc = dataset[col_desc].to_numpy()

  col_corr = []
  for feature in map['corr']:
    col_corr.append(feature['name'])
  X_corr = dataset[col_corr].to_numpy()

  col_sens = []
  for feature in map['sens']:
    col_sens.append(feature['name'])
  X_sens = dataset[col_sens].to_numpy()

  # Target
  Y = dataset[map['target']['name']].to_numpy().reshape(-1, 1)

  ## TRAIN-VAL-TRAIN SPLIT
  # Stratified by target class
  N = X_ind.shape[0]
  indices = np.arange(N)
  train_val_idx, test_index = train_test_split(
      indices,
      test_size=test_size,
      random_state=seed,
      stratify=Y
  )
  train_index, val_index = train_test_split(
      train_val_idx,
      test_size=val_size/(1 - test_size),
      random_state=seed,
      stratify=Y[train_val_idx]
  )

  # Training loader
  train_loader = make_loader(X_ind, X_desc, X_corr, X_sens, Y, train_index, batch_size)

  # Validation loader
  val_loader = make_loader(X_ind, X_desc, X_corr, X_sens, Y, val_index, batch_size)

  # Test loader
  test_loader = make_loader(X_ind, X_desc, X_corr, X_sens, Y, test_index, batch_size)

  return train_loader, val_loader, test_loader