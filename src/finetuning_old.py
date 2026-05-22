import pandas as pd
import os
import gc
import random
import math
from sklearn.utils import resample

from config import Config
from utils import parse_args, load_config, set_global_seeds, setup_logger
from cevaehe.data_loader import make_bucketed_loader
from cevaehe.model import CEVAEHE
from cevaehe.train import train_cevaehe
from cevaehe.test import test_ceveahe
from metrics import get_cca

def main():
  args = parse_args()

  set_global_seeds(args.seed)

  results_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(results_path, exist_ok=True)
  
  # Set up the param search space and how it will be sampled
  search_space = load_config(args.param_space)
  keys = list(search_space.keys())
  values_lists = [search_space[k] for k in keys]
  dimensions = [len(v) for v in values_lists]
  n_configs = math.prod(dimensions)

  n_iterations = min(args.param_iter, n_configs)
  selected_indices = random.sample(range(n_configs), n_iterations)

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  logger.info(f'EXPERIMENT START: {args.exp_name}')
  logger.info(f'Device {args.device}')
  logger.info(f'Data: {args.data}')
  logger.info(f'Mapping: {args.mapping}')
  logger.info(f'Param search space: {args.param_space}')

  logger.info(f'Batch size: {args.batch_size}')
  logger.info(f'Epochs: {args.n_epochs}')
  logger.info(f'VAE learning rate: {args.vae_lr}')
  logger.info(f'Discriminator learning rate: {args.disc_lr}')
  logger.info(f'Discriminator learning period: {args.disc_step}')
  logger.info(f'Distillation Warm-up: {args.distill_warm_up}')
  logger.info(f'TC Loss Warm-up: {args.tc_warm_up}')
  logger.info(f'KL Warm-up: {args.kl_warm_up}')
  logger.info(f'Hidden layers dimension: {args.h_dim}')
  logger.info(f'Activation function: {args.act_fn}')

  all_results = []

  # Load the dataset
  dataset = pd.read_csv(Config.DATA_DIR + args.data)

  # Load the feature mapping
  feature_mapping = load_config(args.mapping)

  for i, idx in enumerate(selected_indices):
    logger.info("="*15)

    current_params = {}
    temp_idx = idx

    for key in reversed(keys):
      vals = search_space[key]
      n = len(vals)
      current_params[key] = vals[temp_idx % n]
      temp_idx //= n

    for param, value in current_params.items():
      setattr(args, param, value)

    def finetuning_run(run_dataset, run_id):
      # Data loaders
      train_loader, val_loader, test_loader = make_bucketed_loader(
        run_dataset, feature_mapping,
        test_size=0.2, batch_size=args.batch_size, seed=args.seed)

      # Feature metadata
      ind_meta = feature_mapping['ind']
      desc_meta = feature_mapping['desc']
      corr_meta = feature_mapping['corr']
      sens_meta = feature_mapping['sens']

      model = CEVAEHE(ind_meta, desc_meta, corr_meta, sens_meta, 
                    args=args)
      
      train_cevaehe(
        model,
        train_loader,
        val_loader,
        logger,
        args
      )

      _, perf_metrics, _ = test_ceveahe(model, test_loader, logger, args)

      res_entry = {"run": f"{i}_{run_id}"} | current_params | perf_metrics
      all_results.append(res_entry)
      
      pd.DataFrame(all_results).to_csv(f"{results_path}/search_results.csv", index=False)

    try:
      for j in range(args.cross_val):
        logger.info(f'Iteration #{i}')
        logger.info(f"Bootstrap #{j}")
        
        boot_dataset = resample(dataset, replace=False, random_state=args.seed + j)

        finetuning_run(boot_dataset, j)

    except Exception as e:
      logger.error(f'Experiment failed: {str(e)}', exc_info=True)

    gc.collect()


  
if __name__ == "__main__":
  main()
  