import argparse
from src.config import Config
import datetime
import os
import json
import logging
import sys

def parse_args():
  date_str = datetime.datetime.now().strftime('%Y-%m-%d')

  parser = argparse.ArgumentParser(description="DCEVAE Training and Testing Pipeline")

  parser.add_argument('--data', type=str)
  parser.add_argument('--mapping', type=str)

  parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
  parser.add_argument('--n_epochs', type=int, default=Config.N_EPOCHS)
  parser.add_argument('--lr', type=int, default=Config.LEARNING_RATE)
  parser.add_argument('--distill_kl_ann', type=int, default=Config.DISTILL_KL_ANN)

  parser.add_argument('--seed', type=int, default=Config.SEED)
  parser.add_argument('--uc_dim', type=int, default=Config.UC_DIM)
  parser.add_argument('--ud_dim', type=int, default=Config.UD_DIM)
  parser.add_argument('--h_dim', type=int, default=Config.H_DIM)
  parser.add_argument('--act_fn', type=str, default=Config.ACT_FN)
  parser.add_argument('--corr_a', type=int, default=Config.CORR_RECON_ALPHA)
  parser.add_argument('--desc_a', type=int, default=Config.DESC_RECON_ALPHA)
  parser.add_argument('--pred_a', type=int, default=Config.PRED_ALPHA)
  parser.add_argument('--fair_b', type=int, default=Config.FAIR_BETA)
  parser.add_argument('--tc_b', type=int, default=Config.TC_BETA)

  parser.add_argument('--exp_name', type=str, default=date_str)

  parser.add_argument('--device', type=str, default=Config.DEVICE)

  return parser.parse_args()

def load_feature_mapping(path):
  if not os.path.exists(path):
    raise FileNotFoundError(f"Mapping file not found at {path}")
    
  with open(path, 'r') as f:
      return json.load(f)
  
def setup_logger(log_dir, exp_name):
  os.makedirs(log_dir, exist_ok=True)
  log_path = os.path.join(log_dir, f"{exp_name}.log")

  # Create a custom logger
  logger = logging.getLogger(exp_name)
  logger.setLevel(logging.INFO)

  if not logger.handlers:
      # Formatter: Timestamp | Level | Message
      formatter = logging.Formatter(
          '%(asctime)s | %(levelname)s | %(message)s', 
          datefmt='%Y-%m-%d %H:%M:%S'
      )

      # File Handler
      file_handler = logging.FileHandler(log_path)
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)

      # Console Handler
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setFormatter(formatter)
      logger.addHandler(console_handler)

  return logger