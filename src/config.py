import torch
import os

class Config:
  DATA_DIR = './z_data/'
  MODELS_DIR = './z_models/'
  LOG_DIR = './z_logs/'
  RESULTS_DIR = './z_results/'

  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  SEED = 4

  # CEVAE-HE architecture
  UC_DIM = 3
  UD_DIM = 3
  H_DIM = 32
  ACT_FN = 'relu'

  # CEVAE-HE training hyperparameters
  BATCH_SIZE = 32
  N_EPOCHS = 5
  VAE_LR = 0.001
  DISC_LR = 0.0001
  DISTILL_WARM_UP = 0
  KL_WARM_UP = 0
  TC_WARM_UP = 0
  CF_INVAR_WARM_UP = 0
  DISC_STEP = 5
  EARLY_STOP_PATIENCE = 10
  EARLY_STOP_START = 50

  # CEVAE-HE loss scaling factors
  DESC_RECON_ALPHA = 1
  PRED_ALPHA = 2
  TC_BETA = 5
  CF_INVAR_BETA = 1
  GROUP_ETA = 2
  GRADNORM_GAMMA = 1.5

  # Classifier training params
  M_SAMPLES = 3
  N_BOOTSTRAPS = 1000

  # SPS params
  SPS_ITER = 20
  SPS_EPOCHS = 100
  