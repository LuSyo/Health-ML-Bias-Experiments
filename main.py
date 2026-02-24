import torch
import pandas as pd
from src.config import Config
from src.data_loader import make_bucketed_loader
from src.model import DCEVAE
from src.train import train_dcevae
from src.test import test_dcevae
from src.utils import parse_args, load_feature_mapping, setup_logger
from src.plots import train_val_loss_curve, disc_tc_loss_curve, distillation_loss_curve

def main():
  args = parse_args()

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  logger.info(f'EXPERIMENT START: {args.exp_name}')
  logger.info(f'Device {args.device}')
  logger.info(f'Data: {args.data}')
  logger.info(f'Mapping: {args.mapping}')
  logger.info(f'Batch size: {args.batch_size}')
  logger.info(f'Epochs: {args.n_epochs}')
  logger.info(f'Learning rate: {args.lr}')
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

    train_val_loss_curve(training_metrics, args)
    disc_tc_loss_curve(training_metrics, args)
    distillation_loss_curve(training_metrics, args)

    test_dcevae(model, test_loader, logger, args)

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)
  
if __name__ == "__main__":
  main()
  