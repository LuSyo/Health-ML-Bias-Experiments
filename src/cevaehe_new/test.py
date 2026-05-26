from typing import cast, List, Any
import torch
from torch import nn
import numpy as np
import pandas as pd
from metrics import get_baseline_bce
from cevaehe_new.causal_validation import run_test_classifier

def test_ceveahe(model, test_loader, logger, args):

  device = args.device
  model.eval()

  all_y_true, all_x_sens, all_u_desc, all_x_desc, all_x_desc_cf, all_y_pred_cf, all_pred_logits =\
    [], [], [], [], [], [], []

  all_x_desc_recon_logits = {}

   # BASELINE Y_RECON_L
  test_y_np = test_loader.dataset.tensors[3].cpu().numpy()
  y_baseline_bce, y_prevalence = get_baseline_bce(test_y_np, weighted=False)
      
  logger.info(f"Test target prevalence: {y_prevalence:.4f}")
  logger.info(f"Test Baseline BCE (y_recon_L ceiling): {y_baseline_bce:.4f}")

  with torch.no_grad():
    for batch in test_loader:
      x_indcorr, x_desc, x_sens, y = [t.to(device) for t in batch[:4]]

      mu_desc, logvar_desc = model.encode(x_desc, x_sens, y=None)

      u_desc = model.reparameterize(mu_desc, logvar_desc)

      x_desc_recon_logits, y_pred_logits, x_desc_recon_cf, y_pred_cf_logits = model.decode(
        u_desc, x_sens
      )

      all_y_true.append(y)
      all_x_sens.append(x_sens)
      all_u_desc.append(u_desc)
      all_x_desc.append(x_desc)
      all_x_desc_cf.append(x_desc_recon_cf)
      all_y_pred_cf.append(torch.sigmoid(y_pred_cf_logits))
      all_pred_logits.append(y_pred_logits)
      
      for feature in model.desc_meta:
        feature_logits = all_x_desc_recon_logits.get(feature['name'], None)
        if feature_logits is not None:
          concat_logits = torch.cat(
            (feature_logits, x_desc_recon_logits[feature['name']]),
            dim=0)
          all_x_desc_recon_logits[feature['name']] = concat_logits
        else:
          all_x_desc_recon_logits[feature['name']] = x_desc_recon_logits[feature['name']]
        
  all_y_true = torch.cat(all_y_true, dim=0)
  all_x_desc = torch.cat(all_x_desc, dim=0)
  all_u_desc = torch.cat(all_u_desc, dim=0)
  all_x_sens = torch.cat(all_x_sens, dim=0)
  all_x_desc_cf = torch.cat(all_x_desc_cf, dim=0)
  all_y_pred_cf = torch.cat(all_y_pred_cf, dim=0)
  all_pred_logits = torch.cat(all_pred_logits, dim=0)
  
  test_outputs = pd.DataFrame({
    'y_true': all_y_true.cpu().numpy().flatten(),
    'x_sens': all_x_sens.cpu().numpy().flatten(),
    'x_desc': list(all_x_desc.cpu().numpy()),
    'u_desc': list(all_u_desc.cpu().numpy()),
    'x_desc_cf': list(all_x_desc_cf.cpu().numpy()),
    'y_pred_cf': all_y_pred_cf.cpu().numpy().flatten(),
  })

  # Reconstruction losses

  desc_recon_L = model.reconstruction_loss(all_x_desc_recon_logits, all_x_desc, model.desc_meta).item()
  y_recon_L = nn.BCEWithLogitsLoss()(all_pred_logits, all_y_true).item()
  
  if len(model.desc_meta) > 0:
    # Latent utility
    mean_x_auprc, std_x_auprc = run_test_classifier(
      features=np.stack(cast(List[Any], test_outputs['x_desc'])),
      target=test_outputs['y_true'],
      scoring="average_precision",
      seed=args.seed
    )

    mean_u_auprc, std_u_auprc = run_test_classifier(
      features=np.stack(cast(List[Any], test_outputs['u_desc'])),
      target=test_outputs['y_true'],
      scoring="average_precision",
      seed=args.seed
    )

    # Latent S info
    mean_x_s_bal_acc, std_x_s_bal_acc = run_test_classifier(
      features=np.stack(cast(List[Any], test_outputs['x_desc'])),
      target=test_outputs['x_sens'],
      scoring="balanced_accuracy",
      seed=args.seed
    )

    mean_u_s_bal_acc, std_u_s_bal_acc = run_test_classifier(
      features=np.stack(cast(List[Any], test_outputs['u_desc'])),
      target=test_outputs['x_sens'],
      scoring="balanced_accuracy",
      seed=args.seed
    )

  else:
    mean_x_auprc = 0
    std_x_auprc = 0
    mean_u_auprc = 0 
    std_u_auprc = 0
    mean_x_s_bal_acc = 0
    std_x_s_bal_acc = 0
    mean_u_s_bal_acc = 0 
    std_u_s_bal_acc = 0

  logger.info("----- TEST RESULTS -----")
  logger.info(f'Reconstruction loss, Xdesc: {desc_recon_L:.4f}')
  logger.info(f'Prediction loss, Y: {y_recon_L:.4f}')
  logger.info(f'Xdesc -> Y, AUPRC: {mean_x_auprc:.4f} (std. {std_x_auprc:.4f})')
  logger.info(f'Udesc -> Y, AUPRC: {mean_u_auprc:.4f} (std. {std_u_auprc:.4f})')
  logger.info(f'Xdesc -> S, Balanced Accuracy: {mean_x_s_bal_acc:.4f} (std. {std_x_s_bal_acc:.4f})')
  logger.info(f'Udesc -> S, Balanced Accuracy: {mean_u_s_bal_acc:.4f} (std. {std_u_s_bal_acc:.4f})')

  perf_metrics = {
    "desc_recon_loss": desc_recon_L,
    "y_recon_loss": y_recon_L,
    "mean_x_auprc": mean_x_auprc,
    "std_x_auprc": std_x_auprc,
    "mean_u_auprc": mean_u_auprc,
    "std_u_auprc": std_u_auprc,
    "mean_x_s_bal_acc": mean_x_s_bal_acc,
    "std_x_s_bal_acc": std_x_s_bal_acc,
    "mean_u_s_bal_acc": mean_u_s_bal_acc,
    "std_u_s_bal_acc": std_u_s_bal_acc
  }

  return test_outputs, perf_metrics

def generate_fair_dataset(model, dataset, feature_mapping, args):
  """
  Generates a CSV of latents and counterfactuals indexed to the original data.
  """
  model.eval()
  device = args.device
  batch_size = args.batch_size

  m_samples = args.m_samples
  ud_dim = model.ud_dim

  col_indcorr = [f['name'] for f in feature_mapping['indcorr']]
  col_desc = [f['name'] for f in feature_mapping['desc']]
  col_sens = [f['name'] for f in feature_mapping['sens']]

  all_counterfactuals: list[pd.DataFrame] = []
  all_latents: list[pd.DataFrame] = []

  with torch.no_grad():
    for i in range(0, len(dataset), batch_size):
      j = min(i+batch_size, len(dataset))
      batch_df = dataset.iloc[i:j]
      B = len(batch_df)

      # Convert to tensors
      x_indcorr = (
        torch.tensor(batch_df[col_indcorr].values, dtype=torch.float32).to(device)
        if col_indcorr else torch.empty((B, 0), device=device)
      )
      x_desc = (
        torch.tensor(batch_df[col_desc].values, dtype=torch.float32).to(device)
        if col_desc else torch.empty((B, 0), device=device)
      )
      x_sens = (torch.tensor(batch_df[col_sens].values, dtype=torch.float32).to(device))

      # Encode features
      mu_desc, logvar_desc = model.encode(x_desc, x_sens, y=None)

      # Sample latents. Shape: (B, m_samples, ud_dim)
      u_desc_samples = model.sample_latent(mu_desc, logvar_desc, m_samples)

      # Flatten batch and sample dims
      u_desc_flat = torch.reshape(u_desc_samples, (B * m_samples, ud_dim))

      # Repeats values in x_sens and x_indcorr m_samples times
      x_sens_flat = torch.repeat_interleave(x_sens, repeats=m_samples, dim=0)
      x_indcorr_flat = torch.repeat_interleave(x_indcorr, repeats=m_samples, dim=0)

      # Decode into counterfactual features and outcome
      _, _, x_desc_recon_cf_flat, y_pred_cf_logits_flat = model.decode(
        u_desc_flat, x_sens_flat
      )

      # Move to CPU and numpy
      u_desc_flat_np = u_desc_flat.cpu().numpy()
      x_desc_cf_np = x_desc_recon_cf_flat.cpu().numpy()
      y_cf_prob_np = torch.sigmoid(y_pred_cf_logits_flat).cpu().numpy()

      # Generate index
      ref_indices = np.repeat(batch_df.index.values, m_samples)
      sample_indices = np.tile(np.arange(m_samples), B)

      # Latent dataframe chunk by chunk
      latent_cols = [f"u_desc_{k}" for k in range(ud_dim)]
      latent_chunk = pd.DataFrame(u_desc_flat_np, columns=latent_cols)
      latent_chunk.insert(0, 'sample_index', sample_indices)
      latent_chunk.insert(0, 'patient_index', ref_indices)
      all_latents.append(latent_chunk)

      # Counterfactual dataframe chunk by chunk
      cf_chunk = pd.DataFrame(x_desc_cf_np, columns=col_desc)
      cf_chunk['y_cf_prob'] = y_cf_prob_np.squeeze()
      cf_chunk.insert(0, 'sample_index', sample_indices)
      cf_chunk.insert(0, 'patient_index', ref_indices)
      all_counterfactuals.append(cf_chunk)

  counterfactuals_df = pd.concat(all_counterfactuals, ignore_index=True)
  latents_df = pd.concat(all_latents, ignore_index=True)

  return counterfactuals_df, latents_df





