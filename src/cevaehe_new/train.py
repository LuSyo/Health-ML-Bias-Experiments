import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
from config import Config
from metrics import get_baseline_bce, compute_group_entropies
from cevaehe_new.model import EarlyStopping

def get_anneal_weight(epoch, warm_up_epochs, loss_weight):
  if epoch < warm_up_epochs:
    return (epoch / warm_up_epochs) * loss_weight
  return loss_weight

def get_mean_grad_norm(model):
  norms = []
  for param in model.parameters():
      if param.grad is not None:
          norms.append(param.grad.norm(2).item())
  return np.mean(norms)

def train_cevaehe(model, train_loader, val_loader, logger, args):
  device = args.device
  model.to(device)
  model = model.train()

  discrim_params = [param for name, param in model.named_parameters() if 'discriminator' in name]
  main_params = [param for name, param in model.named_parameters() if 'discriminator' not in name]
  discrim_optimiser = optim.Adam(discrim_params, lr=args.disc_lr, betas=(0.9, 0.999))
  main_optimiser = optim.Adam(main_params, lr=args.vae_lr, betas=(0.9, 0.999))

  model_path = f'{args.root_dir}{Config.MODELS_DIR}'
  os.makedirs(model_path, exist_ok=True)
  checkpoint_file = f'{model_path}{args.exp_name}_cevaehe.pth'
  torch.serialization.add_safe_globals([argparse.Namespace])

  early_stopping = EarlyStopping(
    patience=args.early_stop_patience, 
    checkpoint_path=checkpoint_file,
    start_epoch=max(args.early_stop_start, args.kl_warm_up, args.tc_warm_up, args.distill_warm_up, args.cf_invar_warm_up),
    max_epoch=args.n_epochs)

  training_log = []
  last_train_results = None

  # BASELINE Y_RECON_L CEILING
  train_y_np = train_loader.dataset.tensors[3].cpu().numpy()
  y_baseline_bce, y_prevalence = get_baseline_bce(train_y_np)
      
  logger.info(f"Training target prevalence: {y_prevalence:.4f}")
  # logger.info(f"Training Baseline Y BCE (y_recon_L ceiling): {y_baseline_bce:.4f}")

  # ADVERSERIAL LOSS ASYMPTOTES
  train_s_np = train_loader.dataset.tensors[2].cpu().numpy()
  disc_chance_floor, s_prevalence = get_baseline_bce(train_s_np, weighted=True)  

  logger.info(f"Training Sensitive Attribute (S) Prevalence: {s_prevalence:.4f}")
  # logger.info(f"Expected VAE TC Loss floor: {tc_chance_floor:.4f}")
  logger.info(f"Expected Discriminator/TC Loss ceiling/floor: {disc_chance_floor:.4f}")
  logger.info(f"Expected Discriminator Balanced Accuracy target: 0.5000")


  # Xdesc group entropy
  train_x_desc = train_loader.dataset.tensors[1].cpu().numpy()
  group_entropies = compute_group_entropies(
    X=train_x_desc,
    x_map=model.desc_meta,
    x_sens=train_s_np,
    n_bootstrap=200,
    seed=args.seed
  )
  desc_entropy_weights = (1 / group_entropies) / np.sum(1 / group_entropies)
  desc_entropy_weights = torch.tensor(desc_entropy_weights, dtype=torch.float32, device=device)

  # pos_weight for the Discriminator loss
  global_num_pos = (train_s_np == 1.0).sum()
  global_num_neg = (train_s_np == 0.0).sum()
  disc_pos_weight = float(global_num_neg / (global_num_pos + 1e-5))

  for epoch in range(args.n_epochs):
    logger.info(f'Epoch {epoch}')

    model.train()

    epoch_metrics = {
      'total_vae_loss': [],
      'desc_recon_L': [],
      'y_recon_L': [],
      'kl_L': [],
      'tc_L': [],
      'cf_invar_L': [],
      'disc_L': [],
      'disc_bal_acc': [],
      'distill_L': []
    }

    for g in range(model.sens_groups):
      epoch_metrics[f'group_{g}_loss'] = []

    epoch_grad_norms = {
      'disc_input_grad_norm': [],
      'disc_output_grad_norm': [],
      'enc_desc_grad_norm': [],
      'enc_corr_grad_norm': []
    }

    all_y_true, all_x_sens, all_u_desc, all_u_desc_inf, all_logvar_desc, all_mu_desc = \
      [], [], [], [], [], []

    distill_weight = get_anneal_weight(epoch, args.distill_warm_up, 1.0)
    kl_weight = get_anneal_weight(epoch, args.kl_warm_up, 1.0)
    tc_weight = get_anneal_weight(epoch, args.tc_warm_up, args.tc_b)
    cf_invar_weight = get_anneal_weight(epoch, args.cf_invar_warm_up, args.cf_invar_b)
    grad_rev_alpha = get_anneal_weight(epoch, args.tc_warm_up, 4.0)

    # Tracking group recon loss across batches
    epoch_strat_recon_losses = torch.zeros(model.sens_groups, device=device)
    loss_ema_gamma = 0.1
    group_eta = args.group_eta

    for i, batch in enumerate(train_loader):
      x_indcorr, x_desc, x_sens, y, x_desc_2, x_sens_2, y_2 =\
      [tensor.to(device) for tensor in batch]

      # INDEPENDENT DISCRIMINATOR UPDATE
      discrim_optimiser.zero_grad()

      if i % args.disc_step == 0:
        with torch.no_grad():
          mu_desc_d, logvar_desc_d = model.encode(x_desc, x_sens, y)
          u_desc_d = model.reparameterize(mu_desc_d, logvar_desc_d)
        
        disc_L, disc_bal_acc = model.disc_loss(u_desc_d, x_sens, pos_weight=disc_pos_weight)
        disc_L.backward()

        # Track discriminator gradients
        disc_input_norm = get_mean_grad_norm(model.discriminator[0])
        epoch_grad_norms['disc_input_grad_norm'].append(disc_input_norm)
        disc_output_norm = get_mean_grad_norm(model.discriminator[-1])
        epoch_grad_norms['disc_output_grad_norm'].append(disc_output_norm)   
        
        discrim_optimiser.step()
      else:
        with torch.no_grad():
          mu_desc_d, logvar_desc_d = model.encode(x_desc, x_sens, y)
          u_desc_d = model.reparameterize(mu_desc_d, logvar_desc_d)
          disc_L, disc_bal_acc = model.disc_loss(u_desc_d, x_sens, pos_weight=disc_pos_weight)

      # MAIN VAE UPDATE
      main_optimiser.zero_grad()

      # Update group weights
      with torch.no_grad():
        group_weights = torch.softmax(group_eta * epoch_strat_recon_losses, dim=0)
      
      vae_outputs = model.calculate_loss(
        x_desc, x_sens, y, 
        x_desc_2, x_sens_2, y_2, 
        distill_weight, kl_weight, tc_weight, cf_invar_weight,
        disc_pos_weight=disc_pos_weight,
        group_weights=group_weights,
        desc_entropy_weights=desc_entropy_weights,
        grad_rev_alpha=grad_rev_alpha
      )

      vae_outputs["total_vae_loss"].backward()

      # Update group loss tracking
      with torch.no_grad():
        for g, g_loss in enumerate(vae_outputs['stratified_y_recon_loss']):
          # Only update weight if instances of the group are present in the batch
          if (x_sens.flatten() == g).sum() > 0:
            epoch_strat_recon_losses[g] = (1 - loss_ema_gamma) * epoch_strat_recon_losses[g] + loss_ema_gamma * g_loss
            epoch_metrics[f"group_{g}_loss"].append(g_loss.item())

      # Track encoder gradients
      enc_desc_norm = get_mean_grad_norm(model.encoder_desc)
      epoch_grad_norms['enc_desc_grad_norm'].append(enc_desc_norm)

      main_optimiser.step()

      # OUTPUTS
      all_y_true.append(y)
      all_x_sens.append(x_sens)
      all_u_desc.append(vae_outputs["u_desc"].detach())
      all_u_desc_inf.append(vae_outputs["u_desc_inf"].detach())
      all_mu_desc.append(vae_outputs["mu_desc"].detach())
      all_logvar_desc.append(vae_outputs["logvar_desc"].detach())

      # LOSSES
      epoch_metrics['total_vae_loss'].append(vae_outputs["total_vae_loss"].item())
      epoch_metrics['desc_recon_L'].append(vae_outputs["desc_recon_L"].item())
      epoch_metrics['y_recon_L'].append(vae_outputs["y_recon_L"].item())
      epoch_metrics['kl_L'].append(vae_outputs["kl_L"].item())
      epoch_metrics['tc_L'].append(vae_outputs["tc_L"].item())
      epoch_metrics['cf_invar_L'].append(vae_outputs["cf_invar_L"].item())
      epoch_metrics['distill_L'].append(vae_outputs["distill_L"].item())
      epoch_metrics['disc_L'].append(disc_L.item())

      # DISC ACCURACY
      epoch_metrics['disc_bal_acc'].append(disc_bal_acc)

    avg_train_loss = np.mean(epoch_metrics['total_vae_loss'])
    training_log.append({'avg_train_loss': avg_train_loss})
    training_log[-1]['avg_y_recon_loss'] = np.mean(epoch_metrics["y_recon_L"])
    training_log[-1]['avg_desc_recon_loss'] = np.mean(epoch_metrics["desc_recon_L"])
    training_log[-1]['avg_disc_loss'] = np.mean(epoch_metrics["disc_L"])
    training_log[-1]['avg_tc_loss'] = np.mean(epoch_metrics["tc_L"])
    training_log[-1]['avg_kl_loss'] = np.mean(epoch_metrics["kl_L"])
    training_log[-1]['avg_cf_invar_loss'] = np.mean(epoch_metrics["cf_invar_L"])
    training_log[-1]['avg_distill_loss'] = np.mean(epoch_metrics["distill_L"])
    training_log[-1]['avg_disc_input_grad'] = np.mean(epoch_grad_norms['disc_input_grad_norm'])
    training_log[-1]['avg_disc_output_grad'] = np.mean(epoch_grad_norms['disc_output_grad_norm'])
    training_log[-1]['avg_desc_grad'] = np.mean(epoch_grad_norms['enc_desc_grad_norm'])
    training_log[-1]['avg_disc_bal_acc'] = np.mean(epoch_metrics["disc_bal_acc"])

    for g in range(model.sens_groups):
      g_list = epoch_metrics[f'group_{g}_loss']
      training_log[-1][f'avg_group_{g}_loss'] = np.mean(g_list) if len(g_list) > 0 else 0.0

    all_y_true = torch.cat(all_y_true, dim=0)
    all_u_desc = torch.cat(all_u_desc, dim=0)
    all_x_sens = torch.cat(all_x_sens, dim=0)
    all_u_desc_inf = torch.cat(all_u_desc_inf, dim=0)
    all_logvar_desc = torch.cat(all_logvar_desc, dim=0)
    all_mu_desc = torch.cat(all_mu_desc, dim=0)

    last_train_results = pd.DataFrame({
      'y_true': all_y_true.cpu().numpy().flatten(),
      'x_sens': all_x_sens.cpu().numpy().flatten(),
      'u_desc_inf': list(all_u_desc_inf.cpu().numpy()),
      'u_desc': list(all_u_desc.cpu().numpy()),
      'logvar_desc': list(all_logvar_desc.cpu().numpy()),
      'mu_desc': list(all_mu_desc.cpu().numpy()),
    })

    # VALIDATION
    model.eval()
    val_y_recon_losses = []
    val_desc_recon_losses = []
    val_tc_losses = []
    val_cf_invar_losses = []
    val_disc_bal_accuracies = []
    val_group_recon_losses = {g: [] for g in range(model.sens_groups)}

    with torch.no_grad():
      for i, batch in enumerate(val_loader):

        _, x_desc, x_sens, y,x_desc_2, x_sens_2, y_2=\
          [tensor.to(device) for tensor in batch]

        vae_outputs = model.calculate_loss(
          x_desc, x_sens, y, 
          x_desc_2, x_sens_2, y_2, 
          distill_weight, kl_weight, tc_weight, cf_invar_weight,
          group_weights=None,
          desc_entropy_weights=desc_entropy_weights
        )

        val_y_recon_losses.append(vae_outputs["y_recon_L"].item())
        val_desc_recon_losses.append(vae_outputs["desc_recon_L"].item())
        val_tc_losses.append(vae_outputs["tc_L"].item())
        val_cf_invar_losses.append(vae_outputs["cf_invar_L"].item())

        _, val_disc_bal_acc = model.disc_loss(
          vae_outputs["u_desc"], 
          x_sens
        )
        val_disc_bal_accuracies.append(val_disc_bal_acc)

        for g, g_loss in enumerate(vae_outputs['stratified_y_recon_loss']):
          if (x_sens.flatten() == g).sum() > 0:
            val_group_recon_losses[g].append(g_loss.item())

    avg_val_tc_loss = np.mean(val_tc_losses)
    avg_val_cf_invar_loss = np.mean(val_cf_invar_losses)
    training_log[-1]['avg_val_y_recon_loss'] = np.mean(val_y_recon_losses)
    training_log[-1]['avg_val_desc_recon_loss'] = np.mean(val_desc_recon_losses)
    training_log[-1]['avg_val_disc_bal_acc'] = np.mean(val_disc_bal_accuracies)

    epoch_val_group_means = {}
    for g in range(model.sens_groups):
      g_val_list = val_group_recon_losses[g]
      g_val_mean = np.mean(g_val_list) if len(g_val_list) > 0 else 0.0
      epoch_val_group_means[g] = g_val_mean
      
      training_log[-1][f'avg_val_group_{g}_loss'] = g_val_mean

    valid_group_losses = [mean for g, mean in epoch_val_group_means.items() if len(val_group_recon_losses[g]) > 0]
    max_avg_val_recon_loss = np.max(valid_group_losses) if valid_group_losses else 0.0

    checkpoint_dict = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_main_state_dict': main_optimiser.state_dict(),
      'discrim_optim_state_dict': discrim_optimiser.state_dict(),
      'args': args
    }

    early_stopping(
      current_recon=max_avg_val_recon_loss, 
      current_tc=avg_val_tc_loss, 
      current_cf_invar=avg_val_cf_invar_loss, 
      current_epoch=epoch,
      checkpoint_dict=checkpoint_dict)
    
    if early_stopping.early_stop:
      logger.info(f"Early stopping triggered at epoch {epoch}")
      break
  
  logger.info(f"Loading best weights from {checkpoint_file}")
  best_state = torch.load(checkpoint_file, weights_only=True)
  model.load_state_dict(best_state['model_state_dict'])
  logger.info(f'model trained for {best_state['epoch'] + 1} epochs')
  
  return training_log, last_train_results