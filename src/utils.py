import argparse
import random
import numpy as np
import torch
from config import Config
import datetime
import os
import json
import logging
import sys

def parse_args():
  date_str = datetime.datetime.now().strftime('%Y-%m-%d')

  parser = argparse.ArgumentParser(description="CEVAE-HE Training and Testing Pipeline")

  # Experiment setup
  parser.add_argument('--exp_name', type=str, default=date_str)
  parser.add_argument('--root_dir', type=str, default='')
  parser.add_argument('--device', type=str, default=Config.DEVICE)
  parser.add_argument('--seed', type=int, default=Config.SEED)
  parser.add_argument('--data', type=str)
  parser.add_argument('--training_data', type=str)
  parser.add_argument('--test_data', type=str)
  parser.add_argument('--mapping', type=str)
  parser.add_argument('--cevaehe', type=str, default=None)
  parser.add_argument('--cf_dataset', type=str, default=None)
  parser.add_argument('--latent_dataset', type=str, default=None)

  # Finetuning 
  parser.add_argument('--param_space', type=str, default=None)
  parser.add_argument('--param_iter', type=int, default=50)
  parser.add_argument('--cross_val', type=int, default=1)

  # SPS audit
  parser.add_argument('--sps_epochs', type=int, default=Config.SPS_EPOCHS)
  parser.add_argument('--sps_iter', type=int, default=Config.SPS_ITER)

  # Training
  parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
  parser.add_argument('--n_epochs', type=int, default=Config.N_EPOCHS)
  parser.add_argument('--vae_lr', type=float, default=Config.VAE_LR)
  parser.add_argument('--disc_lr', type=float, default=Config.DISC_STEP)
  parser.add_argument('--distill_warm_up', type=int, default=Config.DISTILL_WARM_UP)
  parser.add_argument('--kl_warm_up', type=int, default=Config.KL_WARM_UP)
  parser.add_argument('--tc_warm_up', type=int, default=Config.TC_WARM_UP)
  parser.add_argument('--disc_step', type=int, default=Config.DISC_STEP)

  # CEVAE-HE 
  parser.add_argument('--uc_dim', type=int, default=Config.UC_DIM)
  parser.add_argument('--ud_dim', type=int, default=Config.UD_DIM)
  parser.add_argument('--h_dim', type=int, default=Config.H_DIM)
  parser.add_argument('--act_fn', type=str, default=Config.ACT_FN)
  parser.add_argument('--corr_a', type=float, default=Config.CORR_RECON_ALPHA)
  parser.add_argument('--desc_a', type=float, default=Config.DESC_RECON_ALPHA)
  parser.add_argument('--pred_a', type=float, default=Config.PRED_ALPHA)
  parser.add_argument('--fair_b', type=float, default=Config.FAIR_BETA)
  parser.add_argument('--tc_b', type=float, default=Config.TC_BETA)
  parser.add_argument('--u_ind_b', type=float, default=Config.U_IND_BETA)

  # Latent space sampling
  parser.add_argument('--m_samples', type=str, default=Config.M_SAMPLES)

  # Classifier bootstrap training
  parser.add_argument('--n_runs', type=int, default=Config.N_RUNS)

  return parser.parse_args()

def load_config(path):
  if not os.path.exists(path):
    raise FileNotFoundError(f"Config file not found at {path}")
    
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

def set_global_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False