import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
import gc
import time
from config import Config
from metrics import calculate_performance_metrics
from cevaehe.model import EarlyStopping

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

  early_stopping = EarlyStopping(patience=10, checkpoint_path=checkpoint_file, 
                                 start_epoch=max(args.kl_warm_up, args.tc_warm_up, args.distill_warm_up))

  training_log = []
  last_train_results = None

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
      'disc_acc': [],
      'distill_L': []
    }

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

    for i, batch in enumerate(train_loader):
      x_indcorr, x_desc, x_sens, y, x_desc_2, x_sens_2, y_2 =\
      [tensor.to(device) for tensor in batch]

      # Reset optimiser gradients
      discrim_optimiser.zero_grad()
      main_optimiser.zero_grad()

      # Forward pass and loss calculation
      vae_outputs = model.calculate_loss(
        x_indcorr, x_desc, x_sens, y, 
        x_desc_2, x_sens_2, y_2, 
        distill_weight, kl_weight, tc_weight, cf_invar_weight
      )

      # VAE backpropagation
      vae_outputs["total_vae_loss"].backward()

      # Discriminator Loss
      discrim_optimiser.zero_grad()
      if i % args.disc_step == 0:
        disc_L, disc_acc = model.disc_loss(
          vae_outputs["u_desc"], 
          x_sens, 
          vae_outputs["u_desc_2"], 
          x_sens_2
        )

        # Discriminator backpropagation
        disc_L.backward(retain_graph=True)

        # GRADIENTS LOGGING
        disc_input_norm = get_mean_grad_norm(model.discriminator[0])
        epoch_grad_norms['disc_input_grad_norm'].append(disc_input_norm)
        disc_output_norm = get_mean_grad_norm(model.discriminator[-1])
        epoch_grad_norms['disc_output_grad_norm'].append(disc_output_norm)

        discrim_optimiser.step()
      else:
        with torch.no_grad():
          disc_L, disc_acc = model.disc_loss(
              vae_outputs["u_desc"], 
              x_sens, 
              vae_outputs["u_desc_2"], 
              x_sens_2
            )
      
      main_optimiser.step()
      
      # GRADIENTS LOGGING
      enc_desc_norm = get_mean_grad_norm(model.encoder_desc)
      epoch_grad_norms['enc_desc_grad_norm'].append(enc_desc_norm)

      # OUTPUTS
      all_y_true.append(y.cpu().numpy())
      all_x_sens.append(x_sens.cpu().numpy())
      all_u_desc.append(vae_outputs["u_desc"].cpu().numpy())
      all_u_desc_inf.append(vae_outputs["u_desc_inf"].cpu().numpy())
      all_mu_desc.append(vae_outputs["mu_desc"])
      all_logvar_desc.append(vae_outputs["logvar_desc"])

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
      epoch_metrics['disc_acc'].append(disc_acc)

    avg_train_loss = np.mean(epoch_metrics['total_vae_loss'])
    training_log.append({'avg_train_loss': avg_train_loss})
    training_log[-1]['avg_disc_loss'] = np.mean(epoch_metrics["disc_L"])
    training_log[-1]['avg_tc_loss'] = np.mean(epoch_metrics["tc_L"])
    training_log[-1]['avg_kl_loss'] = np.mean(epoch_metrics["kl_L"])
    training_log[-1]['avg_cf_invar_loss'] = np.mean(epoch_metrics["cf_invar_L"])
    training_log[-1]['avg_distill_loss'] = np.mean(epoch_metrics["distill_L"])
    training_log[-1]['avg_desc_recon_loss'] = np.mean(epoch_metrics["desc_recon_L"])
    training_log[-1]['avg_y_recon_loss'] = np.mean(epoch_metrics["y_recon_L"])
    training_log[-1]['avg_disc_input_grad'] = np.mean(epoch_grad_norms['disc_input_grad_norm'])
    training_log[-1]['avg_disc_output_grad'] = np.mean(epoch_grad_norms['disc_output_grad_norm'])
    training_log[-1]['avg_desc_grad'] = np.mean(epoch_grad_norms['enc_desc_grad_norm'])
    training_log[-1]['avg_disc_acc'] = np.mean(epoch_metrics["disc_acc"])

    last_train_results = pd.DataFrame({
      'y_true': np.concatenate(all_y_true).flatten(),
      'x_sens': np.concatenate(all_x_sens).flatten(),
      'u_desc_inf': list(np.concatenate(all_u_desc_inf)),
      'u_desc': list(np.concatenate(all_u_desc)),
      'logvar_desc': list(np.concatenate(all_logvar_desc)),
      'mu_desc': list(np.concatenate(all_mu_desc)),
    })

    # VALIDATION
    model.eval()
    val_recon_losses = []
    val_disc_accuracies = []
    with torch.no_grad():
      for i, batch in enumerate(val_loader):

        x_indcorr, x_desc, x_sens, y,x_desc_2, x_sens_2, y_2=\
          [tensor.to(device) for tensor in batch]

        vae_outputs = model.calculate_loss(
          x_indcorr, x_desc, x_sens, y, 
          x_desc_2, x_sens_2, y_2, 
          distill_weight, kl_weight, tc_weight, cf_invar_weight
        )

        val_recon_losses.append((vae_outputs["desc_recon_L"] + vae_outputs["y_recon_L"]).item())

        _, val_disc_acc = model.disc_loss(
          vae_outputs["u_desc"], 
          x_sens, 
          vae_outputs["u_desc_2"], 
          x_sens_2
        )
        val_disc_accuracies.append(val_disc_acc)

    avg_val_recon_loss = np.mean(val_recon_losses)
    training_log[-1]['avg_val_recon_loss'] = avg_val_recon_loss
    training_log[-1]['avg_val_disc_acc'] = np.mean(val_disc_accuracies)

    checkpoint_dict = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_main_state_dict': main_optimiser.state_dict(),
      'discrim_optim_state_dict': discrim_optimiser.state_dict(),
      'args': args
    }

    # early_stopping(avg_val_recon_loss, training_log[-1]['avg_tc_loss'], checkpoint_dict, epoch)
    
    # if early_stopping.early_stop:
    #   logger.info(f"Early stopping triggered at epoch {epoch}")
    #   break
  
  # logger.info(f"Loading best weights from {checkpoint_file}")
  # best_state = torch.load(checkpoint_file, weights_only=True)
  # model.load_state_dict(best_state['model_state_dict'])
  # logger.info(f'model trained for {best_state['epoch'] + 1} epochs')
  
  return training_log, last_train_results