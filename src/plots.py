import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def apply_gridline_styles(ax):
  ax.minorticks_on()
  ax.tick_params(axis='both', which='minor', color='#999999', grid_color='#999999')
  ax.tick_params(axis='both', which='major', color='#222222', grid_color='#222222')
  ax.grid(visible=True, which='both', axis='both')

def train_val_recon_loss_curve(training_metrics, show=False, sens_groups=2):
  n_epochs = len(training_metrics.index)
  fig, axes = plt.subplots(
    sens_groups + 2, 1, 
    figsize=(0.05 * n_epochs + 1, 4 * (sens_groups + 2)), 
    sharex=True, 
    sharey=True, 
    squeeze=False
  )

  axes = axes.flatten()
  y_lim = 0

  for g in range(sens_groups):
    avg_train_recon_loss = training_metrics[f"avg_group_{g}_y_recon_loss"]
    avg_val_recon_loss = training_metrics[f"avg_val_group_{g}_y_recon_loss"]
    y_lim = max(np.max(avg_train_recon_loss[10:]) + 0.2, np.max(avg_val_recon_loss[10:]) + 0.2, y_lim)

    sns.lineplot(x=training_metrics.index, y=avg_train_recon_loss, ax=axes[g], label='Train Reconstruction Loss', errorbar=None)
    sns.lineplot(x=training_metrics.index+.5, y=avg_val_recon_loss, ax=axes[g], label='Val Reconstruction Loss', errorbar=None)
    axes[g].set_title(f"Sensitive group {g}")
    axes[g].set_ylabel('Total Reconstruction Loss')

    sns.lineplot(x=training_metrics.index, y=avg_train_recon_loss, ax=axes[sens_groups], label=f'Group {g}', errorbar=None)
    sns.lineplot(x=training_metrics.index+.5, y=avg_val_recon_loss, ax=axes[sens_groups + 1], label=f'Group {g}', errorbar=None)
  
  axes[sens_groups].set_title("Training losses")
  axes[sens_groups + 1].set_title("Validation losses")
  for ax in axes:
    apply_gridline_styles(ax)
  plt.ylim(bottom=0, top=y_lim)
  plt.xlabel('Epoch')

  if show: plt.show()

  return fig

def disc_tc_loss_curve(training_metrics, show=False):
  n_epochs = len(training_metrics.index)
  fig, ax = plt.subplots(figsize=(0.05*n_epochs+1, 4))

  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_disc_loss"], ax=ax, label='Discriminator loss', errorbar=None)
  sns.lineplot(x=training_metrics.index, y=training_metrics["avg_tc_loss"], ax=ax, label='VAE TC loss', errorbar=None)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  apply_gridline_styles(ax)

  if show: plt.show()

  return fig

def distillation_loss_curve(training_metrics, show=False):
  n_epochs = len(training_metrics.index)
  fig, ax = plt.subplots(figsize=(0.05*n_epochs+1, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_distill_loss'], ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('Distillation Loss')
  apply_gridline_styles(ax)

  if show: plt.show()

  return fig

def KL_loss_curve(training_metrics, show=False):
  n_epochs = len(training_metrics.index)
  fig, ax = plt.subplots(figsize=(0.05*n_epochs+1, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_kl_loss'], ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('KL Loss')
  apply_gridline_styles(ax)

  if show: plt.show()

  return fig

def all_VAE_losses_curve(training_metrics, show=False):
  n_epochs = len(training_metrics.index)
  fig, axes = plt.subplots(6, 1,figsize=(0.05*n_epochs+1, 20))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_desc_recon_loss'], 
               label="Effective X_desc Recon. Loss", ax=axes[0])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_y_recon_loss'], 
               label="Effective Y Pred. Loss", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_kl_loss'], 
               label="Effective KL Loss", ax=axes[2])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_distill_loss'], 
               label="Effective Distillation Loss", ax=axes[3])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_tc_loss'], 
               label="Effective TC Loss", ax=axes[4])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_cf_invar_loss'], 
               label="Effective Latent CF Invariance Loss", ax=axes[5])
  
  for ax in axes:
    apply_gridline_styles(ax)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  if show: plt.show()

  return fig

def training_accuracy_curve(training_metrics, show=False):
  n_epochs = len(training_metrics.index)
  fig, ax = plt.subplots(figsize=(0.05*n_epochs+1, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['accuracy'], ax=ax)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  apply_gridline_styles(ax)

  if show: plt.show()

  return fig

def disc_acc_train_val_curve(training_metrics, show=False):
  n_epochs = len(training_metrics.index)
  fig, ax = plt.subplots(figsize=(0.05*n_epochs+1, 3))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_disc_bal_acc'], ax=ax, label="Training", errorbar=None)
  sns.lineplot(x=training_metrics.index+.5, y=training_metrics["avg_val_disc_bal_acc"], ax=ax, label='Validation', errorbar=None)
  ax.set_ylim(bottom=0.35, top=0.65)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Discriminator Balanced Accuracy')
  apply_gridline_styles(ax)

  if show: plt.show()

  return fig

def _prepare_for_2d_plot(latent_vectors):
    """Safely maps latent vectors to 2D coordinates, handling 0D and 1D edge cases."""
    if latent_vectors.shape[1] == 0:
        return np.zeros((latent_vectors.shape[0], 2))
    elif latent_vectors.shape[1] == 1:
        jitter = np.random.normal(0, 0.1, size=(latent_vectors.shape[0], 1))
        return np.hstack((latent_vectors, jitter))
    else:
        # Standard behaviour: 2D+ Latent Space
        tsne = TSNE(n_components=2, perplexity=30, random_state=4)
        return tsne.fit_transform(latent_vectors)

def u_clustering_analysis(test_results, mode="test", show=False):
  u_desc = np.stack(test_results['u_desc'].values)
  sens = test_results['x_sens'].values
  y_true = test_results['y_true'].values

  # Dimension reduction with T-distributed Stochastic Neighbor Embedding
  u_desc_2d = _prepare_for_2d_plot(u_desc)

  if mode=="training":
    inf_u_desc = np.stack(test_results['u_desc_inf'].values)
    u_inf_desc_2d = _prepare_for_2d_plot(inf_u_desc)
    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    
    # Row 1, Col 3: Inferred U_desc by sensitive attribute
    sns.scatterplot(x=u_inf_desc_2d[:,0], y=u_inf_desc_2d[:,1], 
                    hue=sens, ax=axes[2], palette='coolwarm')
    axes[2].set_title('Inferred U_desc by Sensitive Attribute')

    # Row 1, Col 4: Inferred U_desc by outcome
    sns.scatterplot(x=u_inf_desc_2d[:,0], y=u_inf_desc_2d[:,1], 
                    hue=y_true, ax=axes[3], palette='viridis')
    axes[3].set_title('Inferred U_desc by Outcome')
  else:
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

  # Row 1, Col 1: U_desc by sensitive attribute
  sns.scatterplot(x=u_desc_2d[:,0], y=u_desc_2d[:,1], 
                  hue=sens, ax=axes[0], palette='coolwarm')
  axes[0].set_title('U_desc by Sensitive Attribute')

  # Row 1, Col 2: U_desc by outcome
  sns.scatterplot(x=u_desc_2d[:,0], y=u_desc_2d[:,1], 
                  hue=y_true, ax=axes[1], palette='viridis')
  axes[1].set_title('U_desc by Outcome')

  if show: plt.show()

  return fig

def grad_curve(training_metrics, show=False):
  fig, axes = plt.subplots(2, 1, figsize=(8, 8))
  ymax_0 = training_metrics['avg_desc_grad'].loc[10:].max(axis=0).max()*1.2
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_desc_grad'], 
               label="Desc encoder average gradient norm", ax=axes[0])
  axes[0].set_ylim(top=ymax_0, bottom=0)
               
  ymax_1 = training_metrics[['avg_disc_input_grad', 'avg_disc_output_grad']].loc[10:].max(axis=0).max()*1.2
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_disc_output_grad'], 
               label="Discriminator last layer average gradient norm", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_disc_input_grad'], 
               label="Discriminator first layer average gradient norm", ax=axes[1])
  axes[1].set_ylim(top=ymax_1, bottom=0)

  for ax in axes:
    apply_gridline_styles(ax)

  plt.xlabel('Epoch')
  plt.ylabel('Gradient Norm')

  if show: plt.show()

  return fig

import matplotlib.pyplot as plt

def stratified_roc_curves(final_curves, models):
    """
    Generates and saves a high-resolution plot comparing stratified 
    average ROC curves for all models.
    """
    mean_fpr = final_curves['mean_fpr']
    colors = {'group_0': 'tab:red', 'group_1': 'tab:blue'} # Female = Red, Male = Blue
    
    # Create a figure with subplots for each model
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    fig.suptitle('Average Stratified ROC Curves across Bootstrap Runs', fontsize=16)

    for i, model in enumerate(models):
      ax = axes[i]
      
      # Plot Female Average Curve
      tpr_f = final_curves[f"{model}_group_0"]
      ax.plot(mean_fpr, tpr_f, color=colors['group_0'], label='Female', linewidth=2)
      
      # Plot Male Average Curve
      tpr_m = final_curves[f"{model}_group_1"]
      ax.plot(mean_fpr, tpr_m, color=colors['group_1'], label='Male', linewidth=2)
      
      # Add reference line
      ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
      
      ax.set_title(f'Model: {model.upper()}')
      ax.set_xlabel('False Positive Rate')
      if i == 0: ax.set_ylabel('True Positive Rate')
      ax.legend(loc='lower right')
      ax.grid(alpha=0.3)

    return fig

