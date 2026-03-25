import torch
import os

class Config:
  DATA_DIR = './data/'
  MODELS_DIR = './models/'
  LOG_DIR = './logs/'
  RESULTS_DIR = './results/'

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
  DISC_STEP = 5

  # CEVAE-HE loss scaling factors
  CORR_RECON_ALPHA = 1
  DESC_RECON_ALPHA = 1
  PRED_ALPHA = 2
  FAIR_BETA = 1
  TC_BETA = 5
  U_IND_BETA = 1

  # Classifier training params
  M_SAMPLES = 3
  N_RUNS = 50

  # SPS params
  SPS_ITER = 20
  SPS_EPOCHS = 100
  