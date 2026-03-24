import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def train_val_loss_curve(training_metrics, show=False):
  fig, ax = plt.subplots(figsize=(8, 4))
  avg_train_recon_loss = training_metrics["avg_corr_recon_loss"]\
                        + training_metrics["avg_desc_recon_loss"]\
                        + training_metrics["avg_y_recon_loss"]
  sns.lineplot(x=training_metrics.index, y=avg_train_recon_loss, ax=ax, label='Train Reconstruction Loss', errorbar=None)
  sns.lineplot(x=training_metrics.index+.5, y=training_metrics["avg_val_recon_loss"], ax=ax, label='Val Reconstruction Loss', errorbar=None)
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Total Reconstruction Loss')

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

  fig, axes = plt.subplots(2, 1,figsize=(8, 8))
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_desc_recon_loss'], 
               label="Effective X_desc Recon. Loss", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_corr_recon_loss'], 
               label="Effective X_corr Recon. Loss", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_y_recon_loss'], 
               label="Effective Y Pred. Loss", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_kl_loss'], 
               label="Effective KL Loss", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_distill_loss'], 
               label="Effective Distillation Loss", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_tc_loss'], 
               label="Effective TC Loss", ax=axes[0])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_fair_loss'], 
               label="Effective Fair Loss", ax=axes[1])
  sns.lineplot(x=training_metrics.index, y=training_metrics['avg_redun_loss'], 
               label="Effective Latent Redundancy Loss", ax=axes[1])
  
  axes[0].minorticks_on()
  axes[0].tick_params(axis='y', which='major', color='#666666')
  axes[0].tick_params(axis='y', which='minor', color='#999999')
  axes[0].grid(visible=True, which='both')
  axes[1].set_ylim(top=15)
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

import matplotlib.pyplot as plt

def stratified_roc_curves(final_curves):
    """
    Generates and saves a high-resolution plot comparing stratified 
    average ROC curves for all models.
    """
    mean_fpr = final_curves['mean_fpr']
    models = ['baseline', 'fair_0', 'fair_1', 'fair_2', 'fair_3']
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

