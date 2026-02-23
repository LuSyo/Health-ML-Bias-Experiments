import torch
import pandas as pd
from config import Config
from src.data_loader import make_bucketed_loader
from src.model import DCEVAE
from src.train import train_dcevae
from src.test import test_dcevae
from src.utils import parse_args, load_feature_mapping, setup_logger

def main():
  args = parse_args()

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  logger.info(f'EXPERIMENT START: {args.exp_name}')
  logger.info(f'Device: {Config.DEVICE}')
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

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)
  
if __name__ == "__main__":
  main()
  