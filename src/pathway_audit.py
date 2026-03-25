import pandas as pd
import numpy as np
import os
import gc

from config import Config
from cevaehe.model import CEVAEHE
from utils import parse_args, load_feature_mapping, set_global_seeds, setup_logger
from cevaehe.causal_validation import run_sps_bootstrap

def main():
  args = parse_args()

  set_global_seeds(args.seed)

  results_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(results_path, exist_ok=True)

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  logger.info(f'PATHWAY SENSITIVY AUDIT START: {args.exp_name}')

  try:
    # Load the dataset
    dataset = pd.read_csv(Config.DATA_DIR + args.data)

    # Load the feature mapping
    feature_mapping = load_feature_mapping(args.mapping)

    # Establish baseline

    baseline_df, audit_df = run_sps_bootstrap(
      dataset,
      feature_mapping,
      iterations=20,
      lite_epochs=100,
      logger=logger,
      args=args,
    )

    baseline_df.to_csv(f'{results_path}/sps_audit_baseline.csv', index=False)
    audit_df.to_csv(f'{results_path}/sps_audit.csv', index=False)


    gc.collect()
  except Exception as e:
    logger.error(f'Audit failed: {str(e)}', exc_info=True)

if __name__ == "__main__":
  main()