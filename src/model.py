import torch
from torch import nn
import numpy as np

class DCEVAE(nn.Module):
  def __init__(self, ind_meta, desc_meta, corr_meta, sens_meta, args):
    super(DCEVAE, self).__init__()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    self.ind_meta = ind_meta
    self.desc_meta = desc_meta
    self.corr_meta = corr_meta
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

    self.ind_dim = get_bucket_dim(ind_meta)
    self.desc_dim = get_bucket_dim(desc_meta)
    self.corr_dim = get_bucket_dim(corr_meta)
    self.sens_dim = get_bucket_dim(sens_meta)
    self.target_dim = 1

    self.args = args
    self.device = args.device
    self.uc_dim = args.uc_dim
    self.ud_dim = args.ud_dim
    self.u_dim = self.uc_dim + self.ud_dim
    self.h_dim = args.h_dim
    self.batch_size = args.batch_size

    # Activation function
    if (args.act_fn == 'relu'):
      self.act_fn = nn.LeakyReLU()
    elif (args.act_fn == 'tanh'):
      self.act_fn = nn.Tanh()

    # Encoder (X_ind, X_desc, X_sens, Y) -> U_d
    input_dim = self.ind_dim + self.desc_dim + self.sens_dim + self.target_dim
    self.encoder_desc = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.mu_desc = nn.Linear(self.h_dim, self.ud_dim)
    self.logvar_desc = nn.Linear(self.h_dim, self.ud_dim)

    # Inference encoder (X_ind, X_desc, X_sens) -> U_d
    input_dim = self.ind_dim + self.desc_dim + self.sens_dim
    self.inf_encoder_desc = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.inf_mu_desc = nn.Linear(self.h_dim, self.ud_dim)
    self.inf_logvar_desc = nn.Linear(self.h_dim, self.ud_dim)

    # Encoder (X_ind, X_corr, X_sens, Y) -> U_c
    input_dim = self.ind_dim + self.corr_dim + self.sens_dim + self.target_dim
    self.encoder_corr = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.mu_corr = nn.Linear(self.h_dim, self.uc_dim)
    self.logvar_corr = nn.Linear(self.h_dim, self.uc_dim)

    # Inference encoder (X_ind, X_corr, X_sens) -> U_c
    input_dim = self.ind_dim + self.corr_dim + self.sens_dim
    self.inf_encoder_corr = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.inf_mu_corr = nn.Linear(self.h_dim, self.uc_dim)
    self.inf_logvar_corr = nn.Linear(self.h_dim, self.uc_dim)

    # Decoder (U_c, X_ind) -> X_corr
    self.decoder_corr = nn.ModuleDict()
    for feature in corr_meta:
      out_dim = 1 if feature['type'] != 'categorical' else feature.get('n')
      self.decoder_corr[feature['name']] = nn.Sequential(
          nn.Linear(self.uc_dim + self.ind_dim, self.h_dim),
          self.act_fn,
          nn.Linear(self.h_dim, out_dim)
      )

    # Decoder (U_desc, X_ind, X_sens) -> X_desc
    self.decoder_desc = nn.ModuleDict()
    for feature in desc_meta:
      out_dim = 1 if feature['type'] != 'categorical' else feature.get('n')
      self.decoder_desc[feature['name']] = nn.Sequential(
          nn.Linear(self.ud_dim + self.ind_dim + self.sens_dim , self.h_dim),
          self.act_fn,
          nn.Linear(self.h_dim, out_dim)
      )

    # Decoder (U_desc, U_corr, X_ind, X_sens) -> Y
    self.decoder_target = nn.Sequential(
        nn.Linear(self.u_dim + self.ind_dim + self.sens_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.target_dim)
    )

    # Discriminator
    self.discriminator = nn.Sequential(
        nn.Linear(self.u_dim + self.sens_dim, self.h_dim),
        nn.LeakyReLU(0.2, True),
        nn.Dropout(p=0.3),
        nn.Linear(self.h_dim, self.h_dim),
        nn.LeakyReLU(0.2, True),
        nn.Dropout(p=0.3),
        nn.Linear(self.h_dim, self.h_dim),
        nn.LeakyReLU(0.2, True),
        nn.Dropout(p=0.3),
        nn.Linear(self.h_dim, 2)
    )

    self.init_params()

  def init_params(self):
    '''
      Initialise the network parameters.
    '''
    main_act_fn = "tanh" if isinstance(self.act_fn, nn.Tanh) else "relu"

    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        if "discriminator" in name:
          # Kaiming normal for the LeakyReLU in the Discriminator
          nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
        else:
          # Xavier normal for Tanh, Kaiming normal for LeakyReLU in the encoders/decoders
          if main_act_fn == "tanh":
            nn.init.xavier_normal(module.weight)
          else:
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")

        if module.bias is not None:
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
    for i, feature in enumerate(x_meta):
      col = x[:, i]
      if feature['type'] == 'categorical':
        processed.append(self.embeddings[feature['name']](col.long()))
      else:
        processed.append(col.unsqueeze(1))
    return torch.cat(processed, dim=1)

  def encode(self, x_ind, x_desc, x_corr, x_sens, y=None):
    '''
      Encoders forward pass
    '''
    # Process raw tensors
    x_ind_p = self._process_features(x_ind, self.ind_meta)
    x_desc_p = self._process_features(x_desc, self.desc_meta)
    x_corr_p = self._process_features(x_corr, self.corr_meta)
    x_sens_p = self._process_features(x_sens, self.sens_meta)

    # Training abduction
    if y is not None:
      # Correlated path encoder
      input_corr = torch.cat((x_ind_p, x_corr_p, x_sens_p, y), dim=1)
      h_corr = self.encoder_corr(input_corr)
      mu_corr = self.mu_corr(h_corr)
      logvar_corr = self.logvar_corr(h_corr)

      # Descendant path encoder
      input_desc = torch.cat((x_ind_p, x_desc_p, x_sens_p, y), dim=1)
      h_desc = self.encoder_desc(input_desc)
      mu_desc = self.mu_desc(h_desc)
      logvar_desc = self.logvar_desc(h_desc)

    else: #Inference
      # Correlated path encoder
      input_corr = torch.cat((x_ind_p, x_corr_p, x_sens_p), dim=1)
      h_corr = self.inf_encoder_corr(input_corr)
      mu_corr = self.inf_mu_corr(h_corr)
      logvar_corr = self.inf_logvar_corr(h_corr)

      # Descendant path encoder
      input_desc = torch.cat((x_ind_p, x_desc_p, x_sens_p), dim=1)
      h_desc = self.inf_encoder_desc(input_desc)
      mu_desc = self.inf_mu_desc(h_desc)
      logvar_desc = self.inf_logvar_desc(h_desc)

    return mu_corr, logvar_corr, mu_desc, logvar_desc

  def decode(self, u_desc, u_corr, x_ind, x_sens):
    '''
      Decoders forward pass
    '''
    # Process raw tensors
    x_ind_p = self._process_features(x_ind, self.ind_meta)
    x_sens_p = self._process_features(x_sens, self.sens_meta)

    # Correlated path
    input_corr = torch.cat((u_corr, x_ind_p), dim=1)
    x_corr_pred = {feature['name']: self.decoder_corr[feature['name']](input_corr)\
                   for feature in self.corr_meta}

    # Descendant path
    input_desc = torch.cat((u_desc, x_ind_p, x_sens_p), dim=1)
    x_desc_pred = {feature['name']: self.decoder_desc[feature['name']](input_desc)\
                   for feature in self.desc_meta}

    # Target
    input_target = torch.cat((u_desc, u_corr, x_ind_p, x_sens_p), dim=1)
    y_pred = self.decoder_target(input_target)

    # Counterfactual
    x_sens_cf_p = self._process_features(1 - x_sens, self.sens_meta)
    input_desc_cf = torch.cat((u_desc, x_ind_p, x_sens_cf_p), dim=1)
    x_desc_cf = {feature['name']: self.decoder_desc[feature['name']](input_desc_cf)\
                   for feature in self.desc_meta}

    input_target_cf = torch.cat((u_desc, u_corr, x_ind_p, x_sens_cf_p), dim=1)
    y_cf = self.decoder_target(input_target_cf)

    return x_corr_pred, x_desc_pred, y_pred, x_desc_cf, y_cf

  def reparameterize(self, mu, logvar):
    '''
      Reparameterisation trick
    '''
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(self.device)
    return mu + eps * std

  def reconstruction_loss(self, v_pred, v, v_meta):
    '''
      Calculates the total reconstruction loss for the given feature bucket,\
       matching each feature type to the correct loss function
    '''
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

  def kl_loss(self, mu, logvar):
    '''
      Calculates the KL divergence for the given latent variable between the \
      approzimate posterior q(u|x,y,a) and the standard normal prior p(u) = N(0,I)
    '''
    logvar_clamped = torch.clamp(logvar, -10.0, 10.0)

    kl_div = -0.5 * torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())

    return kl_div / mu.size(0)

  def kl_divergence(self, mu_fact, logvar_fact, mu_inf, logvar_inf):
    # Clamp logvars to prevent exp() from exploding or precision from becoming inf
    logvar_fact = torch.clamp(logvar_fact, -10, 10)
    logvar_inf = torch.clamp(logvar_inf, -10, 10)

    # Precision of inference (1 / var_inf)
    prec_inf = torch.exp(-logvar_inf)

    # Formula: 0.5 * [log(var_inf/var_fact) + (var_fact + (mu_fact-mu_inf)^2)/var_inf - 1]
    # Broken down into logvar:
    term_log = logvar_inf - logvar_fact
    term_ratio = (torch.exp(logvar_fact) + (mu_fact - mu_inf)**2) * prec_inf

    kl = 0.5 * (term_log + term_ratio - 1.0)

    # Sum over latent dims, then average over batch
    return kl.sum(dim=1).mean()

  def kl_div(self, mu1, logvar1, mu2, logvar2):
    '''
      Calculates the KL divergence between two normal distributions
    '''
    logvar1_clamped = torch.clamp(logvar1, -10.0, 10.0)
    logvar2_clamped = torch.clamp(logvar2, -10.0, 10.0)

    kl_div = 0.5 * torch.sum(logvar2_clamped - logvar1_clamped +
     (torch.exp(logvar1) + (mu1 - mu2)**2)* torch.exp(-logvar2) - 1)

    return kl_div / mu1.size(0)

  def tc_loss(self, u_desc, u_corr, x_sens, u_desc_2, u_corr_2, x_sens_2):
    '''
      Calculates the Total Correlation loss to minimise for the VAE and the \
      discriminator loss
    '''
    sample_size = u_desc.size(0)

    # Process raw tensors
    x_sens_p = self._process_features(x_sens, self.sens_meta)
    x_sens_2_p = self._process_features(x_sens_2, self.sens_meta)

    # Detach U tensors
    u_desc_det = u_desc.detach()
    u_corr_det = u_corr.detach()
    u_desc_2_det = u_desc_2.detach()
    u_corr_2_det = u_corr_2.detach()

    # Run the discriminator on factual samples
    input_disc = torch.cat((u_desc_det, u_corr_det, x_sens_p), dim=1)
    disc_logits = self.discriminator(input_disc) # 1= real samples, 0= permuted samples
    target_real = torch.ones(sample_size, dtype=torch.long).to(self.device)
    target_perm = torch.zeros(sample_size, dtype=torch.long).to(self.device)

    # Prepare U_desc permuted samples from the pre-premuted batch
    permuted_indices = np.random.permutation(u_desc.size(0))
    u_desc_2_permuted = u_desc_2_det[permuted_indices]

    # Run the discriminator on permuted samples
    input_disc_permuted = torch.cat((u_desc_2_permuted, u_corr_2_det, x_sens_2_p), dim=1)
    disc_logits_permuted = self.discriminator(input_disc_permuted)

    # Calculate the TC loss to minimise for the VAE
    # Aim for the discriminator to be wrong for all real samples
    tc_L = nn.CrossEntropyLoss()(disc_logits, target_perm) + nn.CrossEntropyLoss()(disc_logits_permuted, target_real)

    # Calculate the discriminator loss
    disc_L = nn.CrossEntropyLoss()(disc_logits, target_real) + nn.CrossEntropyLoss()(disc_logits_permuted, target_perm)

    return tc_L, disc_L

  def fair_loss(self, y, y_cf):
    y_cf_sig = nn.Sigmoid()(y_cf)
    y_p_sig = nn.Sigmoid()(y)
    fair_L = torch.sum(torch.norm(y_cf_sig - y_p_sig, p=2, dim=1))/y_cf_sig.size(0)
    return fair_L

  def calculate_loss(self, x_ind, x_desc, x_corr, x_sens, y, 
                     x_ind_2, x_desc_2, x_corr_2, x_sens_2, y_2,
                     distill_weight=0, kl_weight=1.0, tc_weight=1.0):

    # Encode
    mu_corr, logvar_corr, mu_desc, logvar_desc = self.encode(
        x_ind, x_desc, x_corr, x_sens, y)
    mu_corr_inf, logvar_corr_inf, mu_desc_inf, logvar_desc_inf = self.encode(
        x_ind, x_desc, x_corr, x_sens, y=None)

    # KL divergence between abduction and inference dists
    distill_L = self.kl_div(mu_corr.detach(), logvar_corr.detach(),
                            mu_corr_inf, logvar_corr_inf) +\
                  self.kl_div(mu_desc.detach(), logvar_desc.detach(),
                              mu_desc_inf, logvar_desc_inf)

    # Reparamaterise
    u_corr = self.reparameterize(mu_corr, logvar_corr)
    u_desc = self.reparameterize(mu_desc, logvar_desc)

    # Decode
    x_corr_pred, x_desc_pred, y_pred, x_desc_cf, y_cf = self.decode(
        u_desc, u_corr, x_ind, x_sens)

    # Reconstruction & prediction loss
    desc_recon_L = self.reconstruction_loss(x_desc_pred, x_desc, self.desc_meta)
    corr_recon_L = self.reconstruction_loss(x_corr_pred, x_corr, self.corr_meta)
    y_recon_L = nn.BCEWithLogitsLoss()(y_pred, y)

    recon_L = self.args.desc_a*desc_recon_L + self.args.corr_a*corr_recon_L + self.args.pred_a*y_recon_L

    # KL loss
    kl_L = self.kl_loss(mu_corr, logvar_corr) + self.kl_loss(mu_desc, logvar_desc)

    # TC loss
    # Pass the permuted batch through the network
    mu_corr_2, logvar_corr_2, mu_desc_2, logvar_desc_2 = self.encode(
        x_ind_2, x_desc_2, x_corr_2, x_sens_2, y_2)
    u_corr_2 = self.reparameterize(mu_corr_2, logvar_corr_2)
    u_desc_2 = self.reparameterize(mu_desc_2, logvar_desc_2)

    tc_L, disc_L = self.tc_loss(u_desc, u_corr, x_sens,
                                u_desc_2, u_corr_2, x_sens_2)

    # Counterfactual fairness loss
    fair_L = self.fair_loss(y_pred, y_cf)

    # Total VAE obective
    # Fair Disentangled Negative ELBO = -M_ELBO + beta_tc * L_TC + beta_f * L_f
    total_vae_loss = recon_L + distill_weight*distill_L + kl_weight*kl_L + tc_weight*tc_L + self.args.fair_b*fair_L

    return total_vae_loss, disc_L, desc_recon_L, corr_recon_L, y_recon_L, kl_L, tc_L, fair_L, distill_L

