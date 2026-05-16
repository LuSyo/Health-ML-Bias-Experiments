import torch
import torch.utils.data as utils
from sklearn.model_selection import train_test_split
import numpy as np

def make_loader(X_indcorr, X_desc, X_sens, Y, index, batch_size=32, seed=4):
  if index is None:
    return None
  
  g = torch.Generator()
  g.manual_seed(seed)
  
  X_indcorr_fact = X_indcorr[index]
  X_desc_fact = X_desc[index]
  X_sens_fact = X_sens[index]
  Y_fact = Y[index]

  # # Permuted set for the discriminator
  permuted_indices = np.random.permutation(index)
  X_desc_perm = X_desc[permuted_indices]
  X_sens_perm = X_sens[permuted_indices]
  Y_perm = Y[permuted_indices]

  X_indcorr_tensor = torch.tensor(X_indcorr_fact, dtype=torch.float32)
  X_desc_tensor = torch.tensor(X_desc_fact, dtype=torch.float32)
  X_sens_tensor = torch.tensor(X_sens_fact, dtype=torch.float32)
  Y_tensor = torch.tensor(Y_fact, dtype=torch.float32)
  X_desc_tensor_2 = torch.tensor(X_desc_perm, dtype=torch.float32)
  X_sens_tensor_2 = torch.tensor(X_sens_perm, dtype=torch.float32)
  Y_tensor_2 = torch.tensor(Y_perm, dtype=torch.float32)

  dataset = utils.TensorDataset(X_indcorr_tensor, X_desc_tensor, X_sens_tensor, Y_tensor, X_desc_tensor_2, X_sens_tensor_2, Y_tensor_2)
  loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

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
      - indcorr_types: a list of the data types of the independent and correlated features
      - desc_types: a list of the data types of the descendant features
      - sens_type: the data type of the sensitive feature
  '''
  np.random.seed(seed=seed)

  ## BUCKET DATASET
  # Independent, Descendant, Correlated features and Sensitive attributes
  col_indcorr = []
  for feature in map['indcorr']:
    col_indcorr.append(feature['name'])
  X_indcorr = dataset[col_indcorr].to_numpy()

  col_desc = []
  for feature in map['desc']:
    col_desc.append(feature['name'])
  X_desc = dataset[col_desc].to_numpy()

  col_sens = []
  for feature in map['sens']:
    col_sens.append(feature['name'])
  X_sens = dataset[col_sens].to_numpy()

  # Target
  Y = dataset[map['target']['name']].to_numpy().reshape(-1, 1)

  ## TRAIN-VAL-TRAIN SPLIT
  # Stratified by target class and sensitive attribute
  strat_cols = col_sens + [map['target']['name']]
  stratify_key = dataset[strat_cols].astype(str).agg('_'.join, axis=1).to_numpy()

  N = X_sens.shape[0]
  indices = np.arange(N)
  if test_size == 0:
    train_val_idx = indices
    test_index = None
  else:
    train_val_idx, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_key
    )

  train_index, val_index = train_test_split(
      train_val_idx,
      test_size=val_size/(1 - test_size),
      random_state=seed,
      stratify=stratify_key[train_val_idx]
  )

  # Training loader
  train_loader = make_loader(X_indcorr, X_desc, X_sens, Y, train_index, batch_size, seed)

  # Validation loader
  val_loader = make_loader(X_indcorr, X_desc, X_sens, Y, val_index, batch_size, seed)

  # Test loader
  test_loader = make_loader(X_indcorr, X_desc, X_sens, Y, test_index, batch_size, seed)

  return train_loader, val_loader, test_loader