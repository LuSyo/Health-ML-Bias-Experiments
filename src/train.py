import numpy as np
import torch
import torch.optim as optim
import os
from src.config import Config

def train_dcevae(model, train_loader, val_loader, logger, args):
  device = args.device
  model.to(device)
  model = model.train()

  discrim_params = [param for name, param in model.named_parameters() if 'discriminator' in name]
  main_params = [param for name, param in model.named_parameters() if 'discriminator' not in name]
  discrim_optimiser = optim.Adam(discrim_params, lr=args.lr)
  main_optimiser = optim.Adam(main_params, lr=args.lr)

  training_log = []
  epoch_metrics_log = []

  for epoch in range(args.n_epochs):
    logger.info(f'--- Start Epoch {epoch}')

    epoch_metrics = {
        'elbo': [],
        'desc_recon_L': [],
        'corr_recon_L': [],
        'y_recon_L': [],
        'tc_L': [],
        'fair_L': [],
        'disc_L': [],
        'distill_L': []
    }

    model.train()
    for i, batch in enumerate(train_loader):
      x_ind, x_desc, x_corr, x_sens, y =\
      [tensor.to(device) for tensor in batch[:5]]

      # Reset optimiser gradients
      discrim_optimiser.zero_grad()
      main_optimiser.zero_grad()

      # Forward pass and loss calculation
      distill_weight = 0 if epoch < args.distill_kl_ann else 1
      elbo, disc_L, desc_recon_L, corr_recon_L, y_recon_L, kl_L, tc_L, fair_L, distill_L \
        = model.calculate_loss(x_ind, x_desc, x_corr, x_sens, y, distill_weight)

      # Discriminator backpropagation
      disc_L.backward(retain_graph=True)

      # Clear VAE optimiser gradient again
      main_optimiser.zero_grad()

      # VAE backpropagation
      elbo.backward()

      # Step both optimisers
      discrim_optimiser.step()
      main_optimiser.step()

      # Log metrics
      epoch_metrics['elbo'].append(elbo.item())
      epoch_metrics['desc_recon_L'].append(desc_recon_L.item())
      epoch_metrics['corr_recon_L'].append(corr_recon_L.item())
      epoch_metrics['y_recon_L'].append(y_recon_L.item())
      epoch_metrics['tc_L'].append(tc_L.item())
      epoch_metrics['fair_L'].append(fair_L.item())
      epoch_metrics['disc_L'].append(disc_L.item())
      epoch_metrics['distill_L'].append(distill_L.item())

    # Epoch summary
    avg_train_loss = np.mean(epoch_metrics['elbo'])
    training_log.append({'avg_train_loss':avg_train_loss})

    training_log[-1]['avg_disc_loss'] = np.mean(epoch_metrics["disc_L"])
    training_log[-1]['avg_tc_loss'] = np.mean(epoch_metrics["tc_L"])
    training_log[-1]['avg_fair_loss'] = np.mean(epoch_metrics["fair_L"])
    training_log[-1]['avg_distill_loss'] = np.mean(epoch_metrics["distill_L"])

    # Validation
    model.eval()
    val_elbo = []
    with torch.no_grad():
      for i, batch in enumerate(val_loader):
        x_ind, x_desc, x_corr, x_sens, y =\
          [tensor.to(device) for tensor in batch[:5]]
        v_elbo, *_ = model.calculate_loss(x_ind, x_desc, x_corr, x_sens, y)
        val_elbo.append(v_elbo.item())

    avg_val_loss = np.mean(val_elbo)
    training_log[-1]['avg_val_loss'] = avg_val_loss

    epoch_metrics_log.append(epoch_metrics)

    logger.info(f'Avg VAE Train Loss: {avg_train_loss}')
    logger.info(f'Avg VAE Validation Loss: {avg_val_loss}')
  
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
  
  return training_log, epoch_metrics_log