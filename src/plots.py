import matplotlib.pyplot as plt
import seaborn as sns

def train_val_loss_curve(training_metrics, show=False):
  fig, ax = plt.subplots(figsize=(8, 4))
  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_train_loss"], ax=ax, label='Train VAE Loss', errorbar=None)
  sns.lineplot(x=training_metrics.index+.5, y=training_metrics["avg_val_loss"], ax=ax, label='Val VAE Loss', errorbar=None)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Total VAE Loss')

  if show: plt.show()

  return fig

def disc_tc_loss_curve(training_metrics, show=False):
  fig, ax = plt.subplots(figsize=(8, 4))
  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_disc_loss"], ax=ax, label='Discriminator loss', errorbar=None)
  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_tc_loss"], ax=ax, label='VAE TC loss', errorbar=None)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  if show: plt.show()

  return fig

def distillation_loss_curve(training_metrics, show=False):
  fig, ax = plt.subplots(figsize=(8, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_distill_loss'], ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('Distillation Loss')

  if show: plt.show()

  return fig

def KL_loss_curve(training_metrics, show=False):
  fig, ax = plt.subplots(figsize=(8, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_kl_loss'], ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('KL Loss')

  if show: plt.show()

  return fig

def all_VAE_losses_curve(training_metrics, show=False):

  fig, ax = plt.subplots(figsize=(8, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_desc_recon_loss'], 
               label="Effective X_desc Recon. Loss", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_corr_recon_loss'], 
               label="Effective X_corr Recon. Loss", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_y_recon_loss'], 
               label="Effective Y Pred. Loss", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_kl_loss'], 
               label="Effective KL Loss", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_distill_loss'], 
               label="Effective Distillation Loss", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_distill_loss'], 
               label="Effective TC Loss", ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  if show: plt.show()

  return fig

def training_accuracy_curve(training_metrics, show=False):
  fig, ax = plt.subplots(figsize=(8, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['accuracy'], ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')

  if show: plt.show()

  return fig