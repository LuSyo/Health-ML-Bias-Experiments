import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.config import Config

def train_val_loss_curve(training_metrics, args, show=False):
  fig, ax = plt.subplots(figsize=(8, 4))
  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_train_loss"], ax=ax, label='Train VAE Loss', errorbar=None)
  sns.lineplot(x=training_metrics.index+.5, y=training_metrics["avg_val_loss"], ax=ax, label='Val VAE Loss', errorbar=None)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Total VAE Loss')
  
  fig_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(fig_path, exist_ok=True)
  plt.savefig(f'{fig_path}/train_val_loss_curve.png')

  if show: plt.show()

def disc_tc_loss_curve(training_metrics, args, show=False):
  fig, ax = plt.subplots(figsize=(8, 4))
  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_disc_loss"], ax=ax, label='Discriminator loss', errorbar=None)
  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_tc_loss"], ax=ax, label='VAE TC loss', errorbar=None)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  fig_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(fig_path, exist_ok=True)
  plt.savefig(f'{fig_path}/disc_tc_loss_curve.png')

  if show: plt.show()

def distillation_loss_curve(training_metrics, args, show=False):
  plt.figure(figsize=(8, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_distill_loss'])
  plt.xlabel('Epoch')
  plt.ylabel('Distillation Loss')

  
  fig_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(fig_path, exist_ok=True)
  plt.savefig(f'{fig_path}/distillation_loss_curve.png')

  if show: plt.show()

