import argparse
import os

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from config import Config
from cevaehe.model import CEVAEHE
from utils import parse_args, load_feature_mapping, set_global_seeds, setup_logger

def main():
  args = parse_args()

  set_global_seeds(args.seed)

  results_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(results_path, exist_ok=True)

  assert args.cevaehe is not None, "Missing CEVAEHE file name"
  assert args.cf_dataset is not None, "Missing counterfactual dataset file path"
  assert args.latent_dataset is not None, "Missing latent spaces dataset file path"

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)
  logger.info(f'EXPERIMENT START: {args.exp_name}')
  logger.info(f'Data: {args.data}')
  logger.info(f'Mapping: {args.mapping}')
  logger.info(f'CEVAEHE Model: {args.cevaehe}')
  logger.info(f'Counterfactual dataset: {args.cf_dataset}')
  logger.info(f'Latent spaces dataset: {args.latent_dataset}')

  try:
    # Load the feature mapping
    feature_mapping = load_feature_mapping(args.mapping)

    # Load the dataset
    dataset = pd.read_csv(Config.DATA_DIR + args.data)

    # Load the latent spaces dataset
    latents_df = pd.read_csv(Config.DATA_DIR + args.latent_dataset)

    # Load the counterfactual dataset
    counterfactuals_df = pd.read_csv(Config.DATA_DIR + args.cf_dataset)

    # Initialise CEVAEHE model and load with trained parameters
    cevaehe = CEVAEHE(feature_mapping['ind'], 
                    feature_mapping['desc'], 
                    feature_mapping['corr'], 
                    feature_mapping['sens'], 
                  args=args)
    model_path = f'{Config.MODELS_DIR}{args.cevaehe}'
    torch.serialization.add_safe_globals([argparse.Namespace])
    model_state = torch.load(model_path, weights_only=True)
    cevaehe.load_state_dict(model_state['model_state_dict']) 


    # Define feature columns to make sure that factual and coutnerfactual datasets
    # have the same column order
    x_ind_col = [f['name'] for f in feature_mapping['ind']]
    x_desc_col = [f['name'] for f in feature_mapping['desc']]
    x_corr_col = [f['name'] for f in feature_mapping['corr']]
    x_sens_col = [f['name'] for f in feature_mapping['sens']]
    feature_cols = x_ind_col + x_desc_col + x_corr_col
    target = feature_mapping['target']['name']

    # Baseline features and target class
    X = dataset[feature_cols]
    X['s_bio'] = dataset[x_sens_col[0]]
    X['s_soc'] = dataset[x_sens_col[0]]
    y = dataset[target]

    # Sociological counterfactual
    X_soc_cf = counterfactuals_df[x_desc_col].merge(dataset[x_ind_col + x_corr_col],
                                                             left_index=True,
                                                             right_index=True)[feature_cols]
    X_soc_cf['s_bio'] = dataset[x_sens_col[0]]
    X_soc_cf['s_soc'] = 1 - dataset[x_sens_col[0]]

    # Merge latents, Xsens and Xind into fair features dataframe
    fair_X = latents_df.merge(dataset[x_ind_col + x_sens_col], right_index=True, left_on='patient_index')

    sss = StratifiedShuffleSplit(n_splits=args.n_runs, test_size=0.2, random_state=args.seed)

    perf_metrics = []

    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
      logger.info(f'--- Start bootstrap loop {i}')
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)

if __name__ == "__main__":
  main()