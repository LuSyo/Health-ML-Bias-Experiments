import pandas as pd
import numpy as np
import os
from src.config import Config
from src.data_loader import make_bucketed_loader
from src.model import DCEVAE
from src.train import train_dcevae
from src.test import test_dcevae
from src.utils import parse_args, load_feature_mapping, setup_logger
from src.plots import train_val_loss_curve, disc_tc_loss_curve, distillation_loss_curve, KL_loss_curve
from src.metrics import get_cca

def main():
  args = parse_args()

  results_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(results_path, exist_ok=True)

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  logger.info(f'EXPERIMENT START: {args.exp_name}')
  logger.info(f'Device {args.device}')
  logger.info(f'Data: {args.data}')
  logger.info(f'Mapping: {args.mapping}')
  logger.info(f'Batch size: {args.batch_size}')
  logger.info(f'Epochs: {args.n_epochs}')
  logger.info(f'VAE learning rate: {args.vae_lr}')
  logger.info(f'Discriminator learning rate: {args.disc_lr}')
  logger.info(f'Distillation KL Annihilation: {args.distill_kl_ann}')
  logger.info(f'U_corr dimension: {args.uc_dim}')
  logger.info(f'U_desc dimension: {args.ud_dim}')
  logger.info(f'Hidden layers dimension: {args.h_dim}')
  logger.info(f'Activation function: {args.act_fn}')
  logger.info(f'Corr. recon. loss alpha: {args.corr_a}')
  logger.info(f'Desc. recon. loss alpha: {args.desc_a}')
  logger.info(f'Prediction loss alpha: {args.pred_a}')
  logger.info(f'Fair loss beta: {args.fair_b}')
  logger.info(f'TC loss beta: {args.tc_b}')

  logger.info('='*30)

  try:
    # Load the dataset
    dataset = pd.read_csv(Config.DATA_DIR + args.data)

    # Load the feature mapping
    feature_mapping = load_feature_mapping(args.mapping)

    # Data loaders
    train_loader, val_loader, test_loader = make_bucketed_loader(dataset, feature_mapping)

    # Feature metadata
    ind_meta = feature_mapping['ind']
    desc_meta = feature_mapping['desc']
    corr_meta = feature_mapping['corr']
    sens_meta = feature_mapping['sens']

    model = DCEVAE(ind_meta, desc_meta, corr_meta, sens_meta, 
                  args=args)
    
    training_log, epoch_metrics_log = train_dcevae(
      model,
      train_loader,
      val_loader,
      logger,
      args
    )

    logger.info(f'END EXPERIMENT {args.exp_name}')  

    training_metrics = pd.DataFrame(training_log)

    train_val_loss_fig = train_val_loss_curve(training_metrics)
    train_val_loss_fig.savefig(f'{results_path}/train_val_loss_curve.png')
    disc_tc_loss_fig = disc_tc_loss_curve(training_metrics)
    disc_tc_loss_fig.savefig(f'{results_path}/disc_tc_loss_curve.png')
    distillation_loss_fig = distillation_loss_curve(training_metrics)
    distillation_loss_fig.savefig(f'{results_path}/distillation_loss_curve.png')
    KL_loss_fig = KL_loss_curve(training_metrics)
    KL_loss_fig.savefig(f'{results_path}/KL_loss_curve.png')

    test_results, perf_metrics, strat_perf_metrics = test_dcevae(model, test_loader, logger, args)

    strat_perf_metrics.to_markdown(f'{results_path}/stratified_perf_metrics.txt', index=False)

    # CAUSAL MODEL VALIDATION
    # Independence of u_desc and u_corr
    u_desc_matrix = np.stack(test_results['u_desc'].values)
    u_corr_matrix = np.stack(test_results['u_corr'].values)
    u_u_cca = get_cca(u_desc_matrix, u_corr_matrix)
    logger.info(f'Canonical Correlation Analysis between U_corr and U_desc: {u_u_cca}')

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)
  
if __name__ == "__main__":
  main()
  