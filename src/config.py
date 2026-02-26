import torch
import os

class Config:
  DATA_DIR = './data/'
  MODELS_DIR = './models/'
  LOG_DIR = './logs/'
  RESULTS_DIR = './results/'

  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  SEED = 4

  # DCEVAE architecture
  UC_DIM = 8
  UD_DIM = 8
  H_DIM = 32
  ACT_FN = 'relu'

  # DCEVAE training hyperparameters
  BATCH_SIZE = 32
  N_EPOCHS = 40
  LEARNING_RATE = 0.01
  DISTILL_KL_ANN = 10

  # DCEVAE loss scaling factors
  CORR_RECON_ALPHA = 1
  DESC_RECON_ALPHA = 1
  PRED_ALPHA = 10
  FAIR_BETA = 1
  TC_BETA = 1