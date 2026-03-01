import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

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
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_tc_loss'], 
               label="Effective TC Loss", ax=ax)
  
  ax.minorticks_on()
  ax.tick_params(axis='y', which='major', color='#666666')
  ax.tick_params(axis='y', which='minor', color='#999999')
  ax.grid(visible=True, which='both')
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

def u_clustering_analysis(test_results, mode="test", show=False):
  u_desc = np.stack(test_results['u_desc'].values)
  u_corr = np.stack(test_results['u_corr'].values)
  sens = test_results['sens'].values
  y_true = test_results['y_true'].values

  # Dimension reduction with T-distributed Stochastic Neighbor Embedding
  tsne = TSNE(n_components=2, perplexity=30, random_state=4)
  u_desc_2d = tsne.fit_transform(u_desc)
  u_corr_2d = tsne.fit_transform(u_corr)

  if mode=="training":
    inf_u_desc = np.stack(test_results['inf_u_desc'].values)
    inf_u_corr = np.stack(test_results['inf_u_corr'].values)
    u_inf_desc_2d = tsne.fit_transform(inf_u_desc)
    u_inf_corr_2d = tsne.fit_transform(inf_u_corr)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1, Col 3: Inferred U_desc by sensitive attribute
    sns.scatterplot(x=u_inf_desc_2d[:,0], y=u_inf_desc_2d[:,1], 
                    hue=sens, ax=axes[0,2], palette='coolwarm')
    axes[0,2].set_title('Inferred U_desc by Sensitive Attribute')

    # Row 1, Col 4: Inferred U_desc by outcome
    sns.scatterplot(x=u_inf_desc_2d[:,0], y=u_inf_desc_2d[:,1], 
                    hue=y_true, ax=axes[0,3], palette='viridis')
    axes[0,3].set_title('Inferred U_desc by Outcome')

    # Row 2, Col 3: Inferred U_corr by sensitive attribute
    sns.scatterplot(x=u_inf_corr_2d[:,0], y=u_inf_corr_2d[:,1], 
                    hue=sens, ax=axes[1,2], palette='coolwarm')
    axes[1,2].set_title('Inferred U_corr by Sensitive Attribute')

    # Row 2, Col 4: Inferred U_corr by outcome
    sns.scatterplot(x=u_inf_corr_2d[:,0], y=u_inf_corr_2d[:,1], 
                    hue=y_true, ax=axes[1,3], palette='viridis')
    axes[1,3].set_title('Inferred U_corr by Outcome')
  else:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

  # Row 1, Col 1: U_desc by sensitive attribute
  sns.scatterplot(x=u_desc_2d[:,0], y=u_desc_2d[:,1], 
                  hue=sens, ax=axes[0,0], palette='coolwarm')
  axes[0,0].set_title('U_desc by Sensitive Attribute')

  # Row 1, Col 2: U_desc by outcome
  sns.scatterplot(x=u_desc_2d[:,0], y=u_desc_2d[:,1], 
                  hue=y_true, ax=axes[0,1], palette='viridis')
  axes[0,1].set_title('U_desc by Outcome')

  # Row 2, Col 1: U_corr by sensitive attribute
  sns.scatterplot(x=u_corr_2d[:,0], y=u_corr_2d[:,1], 
                  hue=sens, ax=axes[1,0], palette='coolwarm')
  axes[1,0].set_title('U_corr by Sensitive Attribute')

  # Row 2, Col 2: U_corr by outcome
  sns.scatterplot(x=u_corr_2d[:,0], y=u_corr_2d[:,1], 
                  hue=y_true, ax=axes[1,1], palette='viridis')
  axes[1,1].set_title('U_corr by Outcome')

  if show: plt.show()

  return fig

def grad_curve(training_metrics, show=False):
  fig, ax = plt.subplots(figsize=(8, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_disc_input_grad'], 
               label="Discriminator first layer average gradient norm", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_disc_output_grad'], 
               label="Discriminator last layer average gradient norm", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_desc_grad'], 
               label="Desc encoder average gradient norm", ax=ax)
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_corr_grad'], 
               label="Corr encoder average gradient norm", ax=ax)
  
  ax.minorticks_on()
  ax.tick_params(axis='y', which='major', color='#666666')
  ax.tick_params(axis='y', which='minor', color='#999999')
  ax.grid(visible=True, which='both')
  
  plt.xlabel('Epoch')
  plt.ylabel('Gradient Norm')

  if show: plt.show()

  return fig

