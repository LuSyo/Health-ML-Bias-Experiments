import math
import torch
from torch import nn
import numpy as np
import copy

class GradientReversal(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, alpha):
      ctx.alpha = alpha
      return x.view_as(x)

  @staticmethod
  def backward(ctx, *grad_outputs):
      # Reverse the gradient direction by multiplying by -alpha
      grad_output = grad_outputs[0]
      grad_input = grad_output.neg() * ctx.alpha
      return grad_input, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class CEVAEHE(nn.Module):
  def __init__(self, desc_meta, sens_meta, args):
    super(CEVAEHE, self).__init__()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    self.desc_meta = desc_meta
    self.sens_meta = sens_meta

    self.embedding_dim = 3
    self.embeddings = nn.ModuleDict()

    def get_bucket_dim(bucket_meta):
      '''
        Returns the total dimension of the feature bucket
      '''
      total_dim = 0
      for feature in bucket_meta:
        if feature['type'] == 'categorical':
          self.embeddings[feature['name']] = nn.Embedding(feature['n'], self.embedding_dim)
          total_dim += self.embedding_dim
        else:
          total_dim += 1
      return total_dim

    self.desc_dim = get_bucket_dim(self.desc_meta)
    self.sens_dim = get_bucket_dim(self.sens_meta)
    self.sens_groups = self.sens_dim if self.sens_meta[0]['type'] == 'categorical' else self.sens_dim + 1
    self.target_dim = 1
    self.args = args
    self.device = args.device
    self.ud_dim = math.ceil(self.desc_dim / 3)
    self.h_dim = args.h_dim
    self.batch_size = args.batch_size

    # Activation function
    if (args.act_fn == 'relu'):
      self.act_fn = nn.LeakyReLU()
    elif (args.act_fn == 'tanh'):
      self.act_fn = nn.Tanh()

    # Adbuction Encoder (Xdesc, X_sens, Y) -> U_d
    input_dim = self.desc_dim + self.sens_dim + self.target_dim
    self.encoder_desc = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.mu_desc = nn.Linear(self.h_dim, self.ud_dim)
    self.logvar_desc = nn.Linear(self.h_dim, self.ud_dim)

    # Inference Encoder (Xdesc, X_sens) -> U_d
    input_dim = self.desc_dim + self.sens_dim
    self.inf_encoder_desc = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.inf_mu_desc = nn.Linear(self.h_dim, self.ud_dim)
    self.inf_logvar_desc = nn.Linear(self.h_dim, self.ud_dim)

    # X_d Decoder (U_d, S_soc) -> X_desc
    self.decoder_desc = nn.ModuleDict()
    for feature in desc_meta:
      out_dim = 1 if feature['type'] != 'categorical' else feature.get('n')
      self.decoder_desc[feature['name']] = nn.Sequential(
        nn.Linear(self.ud_dim + self.sens_dim , self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, out_dim)
      )

    # Y Decoder (U_d, S) -> Y
    self.decoder_target = nn.Sequential(
      nn.Linear(self.ud_dim + self.sens_dim, self.h_dim),
      self.act_fn,
      nn.Linear(self.h_dim, self.target_dim)
    )

    # Discriminator
    self.discriminator = nn.Sequential(
      nn.Linear(self.ud_dim, self.h_dim),
      nn.LeakyReLU(0.2, True),
      nn.Linear(self.h_dim, self.h_dim),
      nn.LeakyReLU(0.2, True),
      nn.Linear(self.h_dim, self.h_dim),
      nn.LeakyReLU(0.2, True),
      nn.Linear(self.h_dim, 1)
    )

    self.init_params()

  def init_params(self):
    '''
      Initialise the network parameters.
    '''
    main_act_fn = "tanh" if isinstance(self.act_fn, nn.Tanh) else "relu"

    for name, module in self.named_modules():      
      if isinstance(module, nn.Linear):
        if module.weight.numel() == 0:
          continue

        if "discriminator" in name:
          # Kaiming normal for the LeakyReLU in the Discriminator
          nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
        else:
          # Xavier normal for Tanh, Kaiming normal for LeakyReLU in the encoders/decoders
          if main_act_fn == "tanh":
            nn.init.xavier_normal(module.weight)
          else:
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")

        if module.bias is not None and module.bias.numel() > 0:
          nn.init.constant_(module.bias, 0)

  def discriminate(self, v):
    '''
      Discriminator forward pass
    '''
    return self.discriminator(v).squeeze()
  
  def _process_features(self, x, x_meta):
    '''
      Processes the input features by applying embeddings to the categorical columns
    '''
    processed = []

    if x is None or len(x_meta) == 0:
      batch_size = x.size(0) if x is not None else 0
      return torch.empty((batch_size, 0), device=self.device)

    for i, feature in enumerate(x_meta):
      col = x[:, i]
      if feature['type'] == 'categorical':
        processed.append(self.embeddings[feature['name']](col.long()))
      else:
        processed.append(col.unsqueeze(1))
    return torch.cat(processed, dim=1)

  def hard_reconstruct_features(self, v_pred, v_meta, batch_size):
    '''
      Reconstructs the feature bucket from its logits
    '''
    
    if len(v_meta) == 0:
      return torch.empty((batch_size, 0), device=self.device)
    
    reconstructed = []
    for i, feature in enumerate(v_meta):
      logits = v_pred[feature['name']]

      if feature['type'] == 'categorical':
        indices = torch.argmax(logits, dim=1).unsqueeze(1)
        reconstructed.append(indices.float())
      elif feature['type'] == 'binary':
        probs = torch.sigmoid(logits)
        binary = (probs > 0.5).float()
        reconstructed.append(binary)
      else:
        reconstructed.append(logits)
    return torch.cat(reconstructed, dim=1)

  def encode(self, x_desc, x_sens, y=None):
    '''
      Encoder forward pass
    '''
    # Process raw tensors
    x_desc_p = self._process_features(x_desc, self.desc_meta)
    x_sens_p = self._process_features(x_sens, self.sens_meta)

    input_desc = torch.cat([t for t in (x_desc_p, x_sens, y) if t is not None], dim=1)

    # Training adbuction
    if y is not None:
      h_desc = self.encoder_desc(input_desc)
      mu_desc = self.mu_desc(h_desc)
      logvar_desc = self.logvar_desc(h_desc)

    else:
      h_desc = self.inf_encoder_desc(input_desc)
      mu_desc = self.inf_mu_desc(h_desc)
      logvar_desc = self.inf_logvar_desc(h_desc)

    return mu_desc, logvar_desc

  def decode(self, u_desc, x_sens):
    '''
      Decoders forward pass

      Outputs:
      - x_desc_recon_logits: the decoded Xdesc as logits
      - y_pred_logits: the decoded outcome as logits
      - x_desc_recon_cf: the decoded counterfactual Xdesc, hard reconstructed
      - y_pred_cf_logits: the decoded counterfactual outcome as logits
    '''

    batch_size = x_sens.size(0)

    # Process raw tensors
    x_sens_p = self._process_features(x_sens, self.sens_meta)

    # Xdesc
    input_desc = torch.cat(
      [t for t in (u_desc, x_sens_p) 
      if t is not None], dim=1
    )
    x_desc_recon_logits = {
      feature['name']: self.decoder_desc[feature['name']](input_desc)\
        for feature in self.desc_meta
    }

    # Target outcome
    input_target = torch.cat(
      [t for t in (u_desc, x_sens_p)
      if t is not None], dim=1
    )
    y_pred_logits = self.decoder_target(input_target)

    # Counterfactual Xdesc
    x_sens_cf_p = self._process_features(1 - x_sens, self.sens_meta)

    input_desc_cf = torch.cat(
      [t for t in (u_desc, x_sens_cf_p) 
      if t is not None], dim=1
    )
    x_desc_cf_logits = {
      feature['name']: self.decoder_desc[feature['name']](input_desc_cf)\
        for feature in self.desc_meta
    }
    x_desc_recon_cf = self.hard_reconstruct_features(
      x_desc_cf_logits, 
      self.desc_meta, batch_size
    )

    # Counterfactual target outcome
    input_target_cf = torch.cat(
      [t for t in (u_desc, x_sens_cf_p)
      if t is not None], dim=1
    )
    y_pred_cf_logits = self.decoder_target(input_target_cf)

    return x_desc_recon_logits, y_pred_logits, x_desc_recon_cf, y_pred_cf_logits

  def reparameterize(self, mu, logvar):
    '''
      Reparameterisation trick
    '''
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(self.device)
    return mu + eps * std

  def sample_latent(self, mu, logvar, m_samples):
    samples = torch.stack([self.reparameterize(mu, logvar) for _ in range(m_samples)], dim=1)

    return samples
  
  def reconstruction_loss(self, v_pred, v, v_meta):
    '''
      Calculates the total reconstruction loss for a feature pathway, matching each feature type to the correct loss function
    '''

    if len(v_meta) == 0:
      return torch.tensor(0.0, device=self.device)
    
    sample_size = v.size(0)
    total_recon_L = 0
    for i, feature in enumerate(v_meta):
      pred = v_pred[feature['name']]
      target = v[:, i]

      if feature['type'] == 'categorical':
        total_recon_L += nn.CrossEntropyLoss(reduction='sum')(pred, target.long())
      elif feature['type'] == 'binary':
        total_recon_L += nn.BCEWithLogitsLoss(reduction='sum')(pred, target.unsqueeze(1))
      else:
        total_recon_L += nn.MSELoss(reduction='sum')(pred, target.unsqueeze(1))

    return total_recon_L / sample_size

  def group_recon_losses(self, x_sens, x_desc, x_desc_recon_logits, y, y_pred_logits, group_weights=None):
    """
    Computes the group-decomposed reconstruction and prediction losses
    """
    sens_flat = x_sens.flatten()
    stratified_recon_losses = torch.zeros(self.sens_groups, device=self.device)
    group_y_recon_losses = torch.zeros(self.sens_groups, device=self.device)
    group_desc_recon_losses = torch.zeros(self.sens_groups, device=self.device)

    for g in range(self.sens_groups):
      mask = (sens_flat == g)

      if mask.sum() == 0:
        continue

      # Subgroup Xdesc reconstruction loss
      g_x_desc_recon_logits = {feat_name: logits[mask] for feat_name, logits in x_desc_recon_logits.items()}
      g_desc_recon = self.reconstruction_loss(g_x_desc_recon_logits, x_desc[mask], self.desc_meta)
      group_desc_recon_losses[g] = g_desc_recon

      # Subgroup prediction loss
      g_y_recon = nn.BCEWithLogitsLoss()(y_pred_logits[mask], y[mask])
      group_y_recon_losses[g] = g_y_recon

      # Total subgroup reconstruction loss
      stratified_recon_losses[g] = self.args.desc_a * g_desc_recon + self.args.pred_a * g_y_recon
    
    y_recon_L = sum(group_y_recon_losses)
    desc_recon_L = sum(group_desc_recon_losses)
    
    if group_weights is not None:
      recon_L = sum([g_weight * g_loss for g_weight, g_loss in zip(group_weights, stratified_recon_losses)])
    else:
      recon_L = stratified_recon_losses.mean()

    return recon_L, y_recon_L, desc_recon_L, stratified_recon_losses

  def kl_loss(self, mu, logvar):
    '''
      Calculates the KL divergence for the given latent variable between the posterior q(u|x,y,a) and the standard normal prior p(u) = N(0,I)
    '''
    logvar_clamped = torch.clamp(logvar, -10.0, 10.0)

    kl_div = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())

    return kl_div / mu.size(0)

  def kl_div(self, mu1, logvar1, mu2, logvar2):
    '''
      Calculates the KL divergence between two normal distributions
    '''
    logvar1_clamped = torch.clamp(logvar1, -10.0, 10.0)
    logvar2_clamped = torch.clamp(logvar2, -10.0, 10.0)

    kl_div = 0.5 * torch.sum(logvar2_clamped - logvar1_clamped +
     (torch.exp(logvar1_clamped) + (mu1 - mu2)**2)* torch.exp(-logvar2_clamped) - 1)

    return kl_div / mu1.size(0)

  def tc_loss(self, u_desc, x_sens):
    """
    Calculates the Total Correlation loss enforcing statistical independence between Udesc and S
    """  
    x_sens = x_sens.squeeze()
    u_desc_reversed = grad_reverse(u_desc, alpha=1.0)
    
    disc_logits = self.discriminate(u_desc_reversed)

    target_sens = x_sens.float().view_as(disc_logits)

    num_pos = (target_sens == 1.0).sum()
    num_neg = (target_sens == 0.0).sum()
    pos_weight = num_neg / (num_pos + 1e-5)

    tc_L = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(disc_logits, target_sens)
    # tc_L = nn.BCEWithLogitsLoss()(disc_logits, target_sens)
    return tc_L

  def disc_loss(self, u_desc, x_sens):
    """
    Calculates the discrimator's loss, training the discriminator to distinguish between real and permuted (Udesc, S) pairs
    """

    ## DISCRIMINATOR FORWARD PASS
    u_desc_det = u_desc.detach()
    disc_logits = self.discriminate(u_desc_det)

    target_sens = x_sens.float().view_as(disc_logits)

    # pos_weight based on X_sens imbalance
    num_pos = (target_sens == 1.0).sum()
    num_neg = (target_sens == 0.0).sum()
    pos_weight = (num_neg / (num_pos + 1e-5))

    # Weighted Loss Function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    disc_L = criterion(disc_logits, target_sens)

    # Tracking Balanced Accuracy
    preds = (disc_logits > 0.0).float()
    tp = ((preds == 1.0) & (target_sens == 1.0)).sum(dim=0).float()
    tn = ((preds == 0.0) & (target_sens == 0.0)).sum(dim=0).float()
    
    sensitivity = tp / (num_pos.item() + 1e-5)
    specificity = tn / (num_neg.item() + 1e-5)
    disc_balanced_acc = ((sensitivity + specificity) / 2.0).mean().item()
    
    return disc_L, disc_balanced_acc


  def calculate_loss(self, x_desc, x_sens, y, x_desc_2, x_sens_2, y_2, distill_weight, kl_weight, tc_weight, cf_invar_weight, group_weights=None):
    '''
      Calculates all components of the VAE loss in training

      Outputs:
        - total_vae_loss: The total VAE loss
        - disc_L: The discriminator loss
        - desc_recon_L: The Xdesc reconstruction loss (in adbuction)
        - y_recon_L: The outcome reconstruction loss (in adbuction)
        - The effective KL loss, including parameterised scaling factor (in adbuction)
        - The effective TC loss, including parameterised scaling factor (in adbuction)
        - The effective Latent CF Invariance loss, including parameterised scaling factor (cycle invariance)
        - The effective distillation loss between adbuction and inference encoders, including parameterised scaling factor
        - y_pred_prob: The predicted outcome (in inference, as probabilities)
        - u_desc: The abducted latent variables (sampled)
        - u_desc_inf: The inferred latent variables (sampled)
        - u_desc_2: The abducted latent on the permuted dataset (sampled)
    '''

    # ENCODING
    mu_desc, logvar_desc = self.encode(
      x_desc, x_sens, y
    )
    mu_desc_inf, logvar_desc_inf = self.encode(
      x_desc, x_sens, y=None
    )

    # DISTILLATION between abduction and inference latent distributions
    distill_L = self.kl_div(
      mu_desc.detach(), logvar_desc.detach(),
      mu_desc_inf, logvar_desc_inf
    )

    # REPARAMERISE
    u_desc = self.reparameterize(mu_desc, logvar_desc)
    u_desc_inf = self.reparameterize(mu_desc_inf, logvar_desc_inf)

    # ABDUCTION DECODING
    x_desc_recon_logits, y_pred_logits, x_desc_recon_cf, y_pred_cf_logits= self.decode(u_desc, x_sens)

    # RECONSTRUCTION AND PREDICTION LOSS
    # groups = torch.unique(x_sens).int()
    # group_desc_recon_losses = []
    # group_y_pred_losses = []
    # for g in groups:
    #   mask = x_sens.flatten() == g
    #   g_desc_recon_L = self.reconstruction_loss(x_desc_recon_logits[mask], x_desc[mask], self.desc_meta)
    #   group_desc_recon_losses.append(g_desc_recon_L)
    #   g_y_pred_L = nn.BCEWithLogitsLoss()(y_pred_logits[mask], y[mask])
    #   group_y_pred_losses.append(g_y_pred_L)

    # desc_recon_L = self.args.desc_a * sum(group_desc_recon_losses)
    # y_recon_L = self.args.pred_a * sum(group_y_pred_losses)
    desc_recon_L = self.reconstruction_loss(x_desc_recon_logits, x_desc, self.desc_meta)
    y_recon_L = nn.BCEWithLogitsLoss()(y_pred_logits, y)
    recon_L = self.args.desc_a * desc_recon_L + self.args.pred_a * y_recon_L

    recon_L, y_recon_L, desc_recon_L, stratified_recon_losses = self.group_recon_losses(
      x_sens,
      x_desc,
      x_desc_recon_logits,
      y,
      y_pred_logits,
      group_weights
    )

    # KL DIVERGENCE
    kl_L = self.kl_loss(mu_desc, logvar_desc)

    # TC LOSS
    tc_L = self.tc_loss(u_desc, x_sens)

    # LATENT COUNTERFACTUAL INVARIANCE LOSS
    x_sens_flipped = 1 - x_sens
    y_pred_cf_prob = torch.sigmoid(y_pred_cf_logits).detach()

    # Cycle invariance: in the counterfactual world previously generated, 
    # Udesc should remain the same
    mu_desc_cf, logvar_desc_cf = self.encode(
      x_desc_recon_cf, x_sens_flipped, y_pred_cf_prob
    )

    cf_invar_L = (
      self.kl_div(mu_desc_cf, logvar_desc_cf, mu_desc, logvar_desc)
      + self.kl_div(mu_desc, logvar_desc, mu_desc_cf, logvar_desc_cf)
    )

    # TOTAL VAE LOSS
    total_vae_loss = (
      recon_L 
      + distill_weight * distill_L
      + kl_weight * kl_L
      + tc_weight * tc_L
      + cf_invar_weight * cf_invar_L
    )

    return {
      "total_vae_loss": total_vae_loss,
      "desc_recon_L": desc_recon_L,
      "y_recon_L": y_recon_L,
      "stratified_recon_losses": stratified_recon_losses,
      "kl_L": kl_weight * kl_L,
      "tc_L": tc_weight * tc_L,
      "cf_invar_L": cf_invar_weight * cf_invar_L,
      "distill_L": distill_weight * distill_L,
      "u_desc": u_desc,
      "mu_desc": mu_desc.detach(),
      "logvar_desc": logvar_desc.detach(),
      "u_desc_inf": u_desc_inf.detach()
    }
  
class EarlyStopping:
  def __init__(self, checkpoint_path, patience=10, min_delta=8e-3, alpha=0.05, start_epoch=10, max_epoch=200):
    self.patience = patience
    self.min_delta = min_delta
    self.checkpoint_path = checkpoint_path
    self.counter = 0
    self.best_recon = None
    self.ema_tc = None # Exponential Moving Average
    self.ema_cf_invar = None # Exponential Moving Average
    self.ema_recon = None # Exponential Moving Average
    self.early_stop = False 
    self.alpha = alpha
    self.start_epoch = start_epoch
    self.max_epoch = max_epoch

  def __call__(self, current_recon, current_tc, current_cf_invar, current_epoch, checkpoint_dict):
    if current_epoch < self.start_epoch:
      return 

    # Priority 1. Monitor stability of TC loss
    if self.ema_tc is None:
        self.ema_tc = current_tc
        tc_delta = 0
        tc_is_stable = False
    else:
        prev_ema_tc = self.ema_tc
        self.ema_tc = (self.alpha * current_tc) + ((1 - self.alpha) * self.ema_tc)
        tc_delta = abs(self.ema_tc - prev_ema_tc) 
        tc_is_stable = tc_delta < self.min_delta

    # Priority 2. Monitor minimal Recon loss
    if self.ema_recon is None:
      self.ema_recon = current_recon
    else:
      self.ema_recon = (self.alpha * current_recon) + ((1 - self.alpha) * self.ema_recon)

    if self.best_recon is None:
        self.best_recon = self.ema_recon
        recon_improved = False
    else:
        recon_improved = self.ema_recon < self.best_recon - self.min_delta

    # Priority 3. Monitor stability of CF invariance loss
    if self.ema_cf_invar is None:
        self.ema_cf_invar = current_cf_invar
        cf_invar_delta = 0
        cf_invar_is_stable = False
    else:
        prev_ema_cf_invar = self.ema_cf_invar
        self.ema_cf_invar = (self.alpha * current_cf_invar) + ((1 - self.alpha) * self.ema_cf_invar)
        cf_invar_delta = abs(self.ema_cf_invar - prev_ema_cf_invar) 
        cf_invar_is_stable = cf_invar_delta < self.min_delta

    # Increment the counter only if recon has not improved AND losses stable
    if not recon_improved and tc_is_stable and cf_invar_is_stable:
      if self.counter == 0:
        self.save_checkpoint(checkpoint_dict)
      self.counter += 1
    else: 
      self.counter = 0
      if recon_improved:
        self.best_recon = self.ema_recon

      if (current_epoch == self.start_epoch) or (current_epoch == self.max_epoch - 1):
        self.save_checkpoint(checkpoint_dict)

    if self.counter >= self.patience:
      self.early_stop = True    

  def save_checkpoint(self, checkpoint_dict):
    frozen_dict = {
      'epoch': checkpoint_dict["epoch"],
      'model_state_dict': {k: v.clone().cpu() for k, v in checkpoint_dict["model_state_dict"].items()}, 
      'optimizer_main_state_dict': copy.deepcopy(checkpoint_dict["optimizer_main_state_dict"]),
      'discrim_optim_state_dict': copy.deepcopy(checkpoint_dict["discrim_optim_state_dict"]),
      'args': checkpoint_dict["args"]
    }
    
    torch.save(frozen_dict, self.checkpoint_path)