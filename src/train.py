import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
from src.config import Config
from src.metrics import calculate_performance_metrics

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

def train_dcevae(model, train_loader, val_loader, logger, args):
  device = args.device
  model.to(device)
  model = model.train()

  discrim_params = [param for name, param in model.named_parameters() if 'discriminator' in name]
  main_params = [param for name, param in model.named_parameters() if 'discriminator' not in name]
  discrim_optimiser = optim.Adam(discrim_params, lr=args.disc_lr)
  main_optimiser = optim.Adam(main_params, lr=args.vae_lr)

  training_log = []
  epoch_metrics_log = []

  for epoch in range(args.n_epochs):
    logger.info(f'--- Start Epoch {epoch}')

    epoch_metrics = {
        'total_vae_loss': [],
        'desc_recon_L': [],
        'corr_recon_L': [],
        'y_recon_L': [],
        'kl_L': [],
        'tc_L': [],
        'fair_L': [],
        'disc_L': [],
        'distill_L': [],
    }

    epoch_grad_norms = {
        'disc_input_grad_norm': [],
        'disc_output_grad_norm': [],
        'enc_desc_grad_norm': [],
        'enc_corr_grad_norm': []
    }

    model.train()
    all_y_true, all_y_pred_prob, all_y_cf_prob, all_sens, all_u_corr, all_u_desc, all_inf_u_corr, all_inf_u_desc = \
    [], [], [], [], [], [], [], []

    for i, batch in enumerate(train_loader):
      x_ind, x_desc, x_corr, x_sens, y, x_ind_2, x_desc_2, x_corr_2, x_sens_2, y_2 =\
      [tensor.to(device) for tensor in batch]

      # Reset optimiser gradients
      discrim_optimiser.zero_grad()
      main_optimiser.zero_grad()

      # Forward pass and loss calculation
      distill_weight = get_anneal_weight(epoch, args.distill_warm_up, 1.0)
      kl_weight = get_anneal_weight(epoch, args.kl_warm_up, 1.0)
      tc_weight = get_anneal_weight(epoch, args.tc_warm_up, args.tc_b)
      total_vae_loss, disc_L, desc_recon_L, corr_recon_L, y_recon_L, kl_L, tc_L, fair_L, distill_L, y_pred_prob, y_cf_prob, inf_u_desc, inf_u_corr, u_desc, u_corr \
        = model.calculate_loss(x_ind, x_desc, x_corr, x_sens, y, 
                               x_ind_2, x_desc_2, x_corr_2, x_sens_2, y_2,
                               distill_weight, kl_weight, tc_weight)

      # Discriminator backpropagation
      disc_L.backward(retain_graph=True)

      # Capture the norms of the discriminator's gradients
      disc_input_norm = get_mean_grad_norm(model.discriminator[0])
      epoch_grad_norms['disc_input_grad_norm'].append(disc_input_norm)
      disc_output_norm = get_mean_grad_norm(model.discriminator[-1])
      epoch_grad_norms['disc_output_grad_norm'].append(disc_output_norm)

      # Clear VAE optimiser gradient again
      main_optimiser.zero_grad()

      # VAE backpropagation
      total_vae_loss.backward()

      # DIAGNOSIS: Capture Encoder gradients here
      enc_desc_norm = get_mean_grad_norm(model.encoder_desc)
      epoch_grad_norms['enc_desc_grad_norm'].append(enc_desc_norm)
      enc_corr_norm = get_mean_grad_norm(model.encoder_corr)
      epoch_grad_norms['enc_corr_grad_norm'].append(enc_corr_norm)

      # Step both optimisers
      discrim_optimiser.step()
      main_optimiser.step()

      all_y_true.append(y.cpu().numpy())
      all_y_pred_prob.append(y_pred_prob.cpu().numpy())
      all_y_cf_prob.append(torch.sigmoid(y_cf_prob).cpu().numpy())
      all_sens.append(x_sens.cpu().numpy())
      all_inf_u_corr.append(inf_u_corr.cpu().numpy())
      all_inf_u_desc.append(inf_u_desc.cpu().numpy())
      all_u_corr.append(u_corr.cpu().numpy())
      all_u_desc.append(u_desc.cpu().numpy())
      
      # Log metrics
      epoch_metrics['total_vae_loss'].append(total_vae_loss.item())
      epoch_metrics['desc_recon_L'].append(desc_recon_L.item())
      epoch_metrics['corr_recon_L'].append(corr_recon_L.item())
      epoch_metrics['y_recon_L'].append(y_recon_L.item())
      epoch_metrics['kl_L'].append(kl_L.item())
      epoch_metrics['tc_L'].append(tc_L.item())
      epoch_metrics['fair_L'].append(fair_L.item())
      epoch_metrics['disc_L'].append(disc_L.item())
      epoch_metrics['distill_L'].append(distill_L.item())

    # Epoch summary
    avg_train_loss = np.mean(epoch_metrics['total_vae_loss'])
    training_log.append({'avg_train_loss': avg_train_loss})

    training_log[-1]['avg_disc_loss'] = np.mean(epoch_metrics["disc_L"])
    training_log[-1]['avg_tc_loss'] = np.mean(epoch_metrics["tc_L"])
    training_log[-1]['avg_kl_loss'] = np.mean(epoch_metrics["kl_L"])
    training_log[-1]['avg_fair_loss'] = np.mean(epoch_metrics["fair_L"])
    training_log[-1]['avg_distill_loss'] = np.mean(epoch_metrics["distill_L"])
    training_log[-1]['avg_desc_recon_loss'] = np.mean(epoch_metrics["desc_recon_L"])
    training_log[-1]['avg_corr_recon_loss'] = np.mean(epoch_metrics["corr_recon_L"])
    training_log[-1]['avg_y_recon_loss'] = np.mean(epoch_metrics["y_recon_L"])
    training_log[-1]['avg_disc_input_grad'] = np.mean(epoch_grad_norms['disc_input_grad_norm'])
    training_log[-1]['avg_disc_output_grad'] = np.mean(epoch_grad_norms['disc_output_grad_norm'])
    training_log[-1]['avg_desc_grad'] = np.mean(epoch_grad_norms['enc_desc_grad_norm'])
    training_log[-1]['avg_corr_grad'] = np.mean(epoch_grad_norms['enc_corr_grad_norm'])

    last_train_results = pd.DataFrame({
      'y_true': np.concatenate(all_y_true).flatten(),
      'y_pred_prob': np.concatenate(all_y_pred_prob).flatten(),
      'y_cf_prob': np.concatenate(all_y_cf_prob).flatten(),
      'sens': np.concatenate(all_sens).flatten(),
      'inf_u_corr': list(np.concatenate(all_inf_u_corr)),
      'inf_u_desc': list(np.concatenate(all_inf_u_desc)),
      'u_corr': list(np.concatenate(all_u_corr)),
      'u_desc': list(np.concatenate(all_u_desc)),
    })
    last_train_results['y_pred'] = (last_train_results['y_pred_prob'] > 0.5).astype(int)

    perf_metrics = calculate_performance_metrics(
      last_train_results['y_true'],
      last_train_results['y_pred'],
      last_train_results['y_pred_prob'])
    
    training_log[-1]['accuracy'] = perf_metrics['accuracy']

    # Validation
    model.eval()
    val_vae_loss = []
    with torch.no_grad():
      for i, batch in enumerate(val_loader):
        x_ind, x_desc, x_corr, x_sens, y, x_ind_2, x_desc_2, x_corr_2, x_sens_2, y_2=\
          [tensor.to(device) for tensor in batch]
        v_vae_loss, *_ = model.calculate_loss(x_ind, x_desc, x_corr, x_sens, y,
                                              x_ind_2, x_desc_2, x_corr_2, x_sens_2, y_2)
        val_vae_loss.append(v_vae_loss.item())

    avg_val_loss = np.mean(val_vae_loss)
    training_log[-1]['avg_val_loss'] = avg_val_loss

    epoch_metrics_log.append(epoch_metrics)

    logger.info(f'Avg VAE Train Loss: {avg_train_loss}')
    logger.info(f'Avg VAE Validation Loss: {avg_val_loss}')
    logger.info(f'Training accuracy: {perf_metrics['accuracy']}')
  
  model_path = f'{args.root_dir}{Config.MODELS_DIR}'
  os.makedirs(model_path, exist_ok=True)
  torch.save({
    'epoch': args.n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_main_state_dict': main_optimiser.state_dict(),
    'discrim_optim_state_dict': discrim_optimiser.state_dict(),
    'args': args
  }, f'{model_path}{args.exp_name}_dcevae.pth')
  logger.info(f'DCEVAE model saved to {model_path}')
  
  return training_log, epoch_metrics_log, last_train_results