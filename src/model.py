import torch
from torch import nn
import numpy as np

class CEVAEHE(nn.Module):
  def __init__(self, ind_meta, desc_meta, corr_meta, sens_meta, args):
    super(CEVAEHE, self).__init__()
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

    # Encoder (Xdesc, Xind, Y) -> U_d
    input_dim = self.desc_dim + self.ind_dim + self.target_dim
    self.encoder_desc = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.mu_desc = nn.Linear(self.h_dim, self.ud_dim)
    self.logvar_desc = nn.Linear(self.h_dim, self.ud_dim)

    # Inference encoder (Xdesc, Xind) -> U_d
    input_dim = self.desc_dim + self.ind_dim
    self.inf_encoder_desc = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.inf_mu_desc = nn.Linear(self.h_dim, self.ud_dim)
    self.inf_logvar_desc = nn.Linear(self.h_dim, self.ud_dim)

    # Encoder (X_corr, Xind, X_sens, Y) -> U_c
    input_dim = self.corr_dim + self.ind_dim + self.sens_dim + self.target_dim
    self.encoder_corr = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.mu_corr = nn.Linear(self.h_dim, self.uc_dim)
    self.logvar_corr = nn.Linear(self.h_dim, self.uc_dim)

    # Inference encoder (X_corr, Xind, X_sens) -> U_c
    input_dim = self.corr_dim + self.ind_dim + self.sens_dim
    self.inf_encoder_corr = nn.Sequential(
        nn.Linear(input_dim, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.h_dim),
        self.act_fn
    )
    self.inf_mu_corr = nn.Linear(self.h_dim, self.uc_dim)
    self.inf_logvar_corr = nn.Linear(self.h_dim, self.uc_dim)

    # Decoder (U_c, X_ind, Sbio) -> X_corr
    self.decoder_corr = nn.ModuleDict()
    for feature in corr_meta:
      out_dim = 1 if feature['type'] != 'categorical' else feature.get('n')
      self.decoder_corr[feature['name']] = nn.Sequential(
          nn.Linear(self.uc_dim + self.ind_dim + self.sens_dim, self.h_dim),
          self.act_fn,
          nn.Linear(self.h_dim, out_dim)
      )

    # Decoder (U_desc, X_ind, Ssoc) -> X_desc
    self.decoder_desc = nn.ModuleDict()
    for feature in desc_meta:
      out_dim = 1 if feature['type'] != 'categorical' else feature.get('n')
      self.decoder_desc[feature['name']] = nn.Sequential(
          nn.Linear(self.ud_dim + self.ind_dim + self.sens_dim , self.h_dim),
          self.act_fn,
          nn.Linear(self.h_dim, out_dim)
      )

    # Decoder (U_desc, U_corr, X_ind, Sbio/Ssoc) -> Y
    self.decoder_target = nn.Sequential(
        nn.Linear(self.u_dim + self.ind_dim + self.sens_dim*2, self.h_dim),
        self.act_fn,
        nn.Linear(self.h_dim, self.target_dim)
    )

    # Discriminator
    self.discriminator = nn.Sequential(
        nn.Linear(self.ud_dim + self.sens_dim, self.h_dim),
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

  def encode(self, x_desc, x_corr, x_ind, sens_bio, y=None):
    '''
      Encoders forward pass
    '''
    # Process raw tensors
    x_desc_p = self._process_features(x_desc, self.desc_meta)
    x_corr_p = self._process_features(x_corr, self.corr_meta)
    x_ind_p = self._process_features(x_ind, self.ind_meta)
    sens_bio_p = self._process_features(sens_bio, self.sens_meta)

    # Training abduction
    if y is not None:
      # Correlated path encoder
      input_corr = torch.cat((x_corr_p, x_ind_p, sens_bio_p, y), dim=1)
      h_corr = self.encoder_corr(input_corr)
      mu_corr = self.mu_corr(h_corr)
      logvar_corr = self.logvar_corr(h_corr)

      # Descendant path encoder
      input_desc = torch.cat((x_desc_p, x_ind_p, y), dim=1)
      h_desc = self.encoder_desc(input_desc)
      mu_desc = self.mu_desc(h_desc)
      logvar_desc = self.logvar_desc(h_desc)

    else: #Inference
      # Correlated path encoder
      input_corr = torch.cat((x_corr_p, x_ind_p, sens_bio_p), dim=1)
      h_corr = self.inf_encoder_corr(input_corr)
      mu_corr = self.inf_mu_corr(h_corr)
      logvar_corr = self.inf_logvar_corr(h_corr)

      # Descendant path encoder
      input_desc = torch.cat((x_desc_p, x_ind_p), dim=1)
      h_desc = self.inf_encoder_desc(input_desc)
      mu_desc = self.inf_mu_desc(h_desc)
      logvar_desc = self.inf_logvar_desc(h_desc)

    return mu_corr, logvar_corr, mu_desc, logvar_desc

  def decode(self, u_desc, u_corr, x_ind, sens_bio, sens_soc):
    '''
      Decoders forward pass

      Outputs:
        - x_desc_pred_logits: the decoded Xdesc (as logits)
        - x_corr_pred_logits: the decoded Xcorr (as logits)
        - y_pred_logits: the decoded outcome (as logits)
        - x_desc_cf: the decoded soc-counterfactual Xdesc (hard reconstructed) 
        - x_corr_cf: the decoded bio-counterfactual Xcorr (hard reconstructed) 
        - y_soc_cf_logits: the decoded soc-counterfactual outcome (as logits)
        - y_bio_cf_logits: the decoded bio-counterfactual outcome (as logits)
        - y_full_cf_logits: the decoded full-counterfactual outcome (as logits)
    '''
    # Process raw tensors
    x_ind_p = self._process_features(x_ind, self.ind_meta)
    sens_bio_p = self._process_features(sens_bio, self.sens_meta)
    sens_soc_p = self._process_features(sens_soc, self.sens_meta)

    # Correlated path
    input_corr = torch.cat((u_corr, x_ind_p, sens_bio_p), dim=1)
    x_corr_pred_logits = {feature['name']: self.decoder_corr[feature['name']](input_corr)\
                   for feature in self.corr_meta}

    # Descendant path
    input_desc = torch.cat((u_desc, x_ind_p, sens_soc_p), dim=1)
    x_desc_pred_logits = {feature['name']: self.decoder_desc[feature['name']](input_desc)\
                   for feature in self.desc_meta}

    # Target
    input_target = torch.cat((u_desc, u_corr, x_ind_p, 
                              sens_bio_p, sens_soc_p), dim=1)
    y_pred_logits = self.decoder_target(input_target)

    # Sociological Counterfactual
    sens_soc_cf_p = self._process_features(1 - sens_soc, self.sens_meta)
    input_desc_cf = torch.cat((u_desc, x_ind_p, sens_soc_cf_p), dim=1)
    x_desc_cf_logits = {feature['name']: 
                        self.decoder_desc[feature['name']](input_desc_cf)\
                   for feature in self.desc_meta}
    x_desc_cf = self.hard_reconstruct_features(x_desc_cf_logits, self.desc_meta)

    input_target_soc_cf = torch.cat((u_desc, u_corr, x_ind_p, 
                                 sens_bio_p, sens_soc_cf_p), dim=1)
    y_soc_cf_logits = self.decoder_target(input_target_soc_cf)

    # Biological Counterfactual
    sens_bio_cf_p = self._process_features(1 - sens_bio, self.sens_meta)
    input_corr_cf = torch.cat((u_corr, x_ind_p, sens_bio_cf_p), dim=1)
    x_corr_cf_logits = {feature['name']: 
                        self.decoder_corr[feature['name']](input_corr_cf)\
                   for feature in self.corr_meta}
    x_corr_cf = self.hard_reconstruct_features(x_corr_cf_logits, self.corr_meta)

    input_target_bio_cf = torch.cat((u_desc, u_corr, x_ind_p, 
                                 sens_bio_cf_p, sens_soc_p), dim=1)
    y_bio_cf_logits = self.decoder_target(input_target_bio_cf)

    # Full Counterfactual?
    input_target_full_cf = torch.cat((u_desc, u_corr, x_ind_p,
                                      sens_bio_cf_p, sens_soc_cf_p), dim=1)
    y_full_cf_logits = self.decoder_target(input_target_full_cf)

    return x_desc_pred_logits, x_corr_pred_logits, y_pred_logits, x_desc_cf, x_corr_cf, y_soc_cf_logits, y_bio_cf_logits, y_full_cf_logits

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
  
  def hard_reconstruct_features(self, v_pred, v_meta):
    '''
      Reconstructs the feature bucket from its logits
    '''
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


  def kl_loss(self, mu, logvar):
    '''
      Calculates the KL divergence for the given latent variable between the \
      approzimate posterior q(u|x,y,a) and the standard normal prior p(u) = N(0,I)
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
  
  def tc_loss(self, u_desc, s_soc):
    sample_size = u_desc.size(0)

    # Discriminator prediction targets
    target_real = torch.ones(sample_size, dtype=torch.long).to(self.device)
    target_flipped = torch.zeros(sample_size, dtype=torch.long).to(self.device)

    # process sensitive feature
    s_soc_p = self._process_features(s_soc, self.sens_meta)
    s_soc_flipped_p = self._process_features(1- s_soc, self.sens_meta)

    ## VAE LOSS
    input_disc_att = torch.cat((u_desc, s_soc_p), dim=1)
    disc_logits_real_att = self.discriminate(input_disc_att)

    input_disc_flipped_att = torch.cat((u_desc, s_soc_flipped_p), dim=1)
    disc_logits_flipped_att = self.discriminate(input_disc_flipped_att)

    tc_L = nn.CrossEntropyLoss()(disc_logits_real_att, target_flipped)\
          + nn.CrossEntropyLoss()(disc_logits_flipped_att, target_real)
    
    return tc_L
  
  def disc_loss(self, u_desc, s_soc):
    sample_size = u_desc.size(0)

    # Discriminator prediction targets
    target_real = torch.ones(sample_size, dtype=torch.long).to(self.device)
    target_flipped = torch.zeros(sample_size, dtype=torch.long).to(self.device)

    # process sensitive feature
    s_soc_p = self._process_features(s_soc, self.sens_meta)
    s_soc_flipped_p = self._process_features(1- s_soc, self.sens_meta)

    ## DISCRIMINATOR TRAINING
    u_desc_det = u_desc.detach()

    input_disc_det = torch.cat((u_desc_det, s_soc_p), dim=1)
    disc_logits_real = self.discriminate(input_disc_det)

    input_disc_flipped = torch.cat((u_desc_det, s_soc_flipped_p), dim=1)
    disc_logits_flipped = self.discriminate(input_disc_flipped)

    # Discriminator Loss
    disc_L = nn.CrossEntropyLoss()(disc_logits_real, target_real)\
            + nn.CrossEntropyLoss()(disc_logits_flipped, target_flipped)
    
    return disc_L

  def u_redundancy_loss(self, u_desc, u_corr, x_ind, s_soc, s_bio, y_pred_prob):
    '''
      Calculates the Latent redundancy loss to minimise the predictive information shared by Udesc and Ucorr
    '''
    # Process raw tensors
    s_soc_p = self._process_features(s_soc, self.sens_meta)
    s_bio_p = self._process_features(s_bio, self.sens_meta)
    x_ind_p = self._process_features(x_ind, self.ind_meta)

    # Create permuted u_corr and reconstructed outcomes
    permuted_indices = np.random.permutation(u_corr.size(0))
    u_corr_permuted = u_corr[permuted_indices]

    for param in self.decoder_target.parameters():
        param.requires_grad = False

    # Forward target decoder pass on permuted data
    input_permuted = torch.cat((u_desc, u_corr_permuted, x_ind_p, 
                            s_bio_p, s_soc_p), dim=1)
    y_permuted_logits = self.decoder_target(input_permuted)

    for param in self.decoder_target.parameters():
        param.requires_grad = True
    
    u_redun_L = nn.BCEWithLogitsLoss()(y_permuted_logits, y_pred_prob.detach())

    return u_redun_L

  def fair_loss(self, y_true, y_cf, y_pred, y_pred_cf):
    '''
      Fairness loss, implements the Individual Equalised Counterfactual Odds criteria for counterfactual fairness:
      An individual whose sensitive attribute is changed in a counterfactual world, all independent factors being the same, should receive the same prediction as their image, given that they have the same actual outcome. And they should receive a different prediction if they have a different outcome. (=> equality of error rates across counterfactuals)

      Inputs:
        - y_true: The factual actual outcome (as binary)
        - y_cf: The counterfactual actual outcome (as probabilities)
        - y_pred: The factual prediction (as probabilities)
        - y_pred_cf: The counterfactual prediction (as probabilities)

      Outputs:
        fair_L: the mean squared error (MSE) between factual and counterfactual prediction, conditioned on equal outcomes, added to the MSE between opposite factual and counterfactual, conditioned on opposite outcomes
    '''
    y_cf_hard = (y_cf > 0.5).float()
    equal_outcome = y_cf_hard == y_true
    opp_outcome = y_cf_hard == 1 - y_true

    if not equal_outcome.any():
      fair_L_equal = torch.tensor(0.0, device=self.device, requires_grad=True)
    else:
      y_pred_equal = y_pred[equal_outcome]
      y_pred_cf_equal = y_pred_cf[equal_outcome]
      fair_L_equal = nn.MSELoss()(y_pred_cf_equal, y_pred_equal)

    if not opp_outcome.any():
        fair_L_opp = torch.tensor(0.0, device=self.device, requires_grad=True)
    else:
      y_pred_opp = y_pred[opp_outcome]
      y_pred_cf_opp = y_pred_cf[opp_outcome]
      fair_L_opp = nn.MSELoss()(y_pred_cf_opp, 1 - y_pred_opp)

    return fair_L_equal + fair_L_opp
  
  def calculate_loss(self, x_ind, x_desc, x_corr, x_sens, y, 
                     x_ind_2, x_desc_2, x_corr_2, x_sens_2, y_2,
                     distill_weight=0, kl_weight=1.0, tc_weight=1.0):
    '''
      Calculates all components of the VAE loss in training

      Outputs:
        - total_vae_loss: The total VAE loss
        - disc_L: The discriminator loss
        - desc_recon_L: The Xdesc reconstruction loss (in adbuction)
        - corr_recon_L: The Xcorr reconstruction loss (in adbuction)
        - y_recon_L: The outcome recosntruction loss (in adbuction)
        - The effective KL loss, including parameterised scaling factor (in adbuction)
        - The effective TC loss, including parameterised scaling factor (in adbuction)
        - The effective Fair loss, including parameterised scaling factor (in inference)
        - The effective distillation loss between adbuction and inference encoders, including parameterised scaling factor
        - y_pred_prob: The predicted outcome (in inference, as probabilities)
        - mu_desc, mu_corr: The abducted latent variables (mean)
        - mu_desc_inf, mu_corr_inf: The inferred latent variables (mean)

    '''

    # Split the sensitive attribute
    s_soc = x_sens.clone()
    s_bio = x_sens.clone()

    # Encode
    mu_corr, logvar_corr, mu_desc, logvar_desc = self.encode(
        x_desc, x_corr, x_ind, s_bio, y)
    mu_corr_inf, logvar_corr_inf, mu_desc_inf, logvar_desc_inf = self.encode(
        x_desc, x_corr, x_ind, s_bio, y=None)

    # KL divergence between abduction and inference dists
    distill_L = self.kl_div(mu_corr.detach(), logvar_corr.detach(),
                            mu_corr_inf, logvar_corr_inf) +\
                  self.kl_div(mu_desc.detach(), logvar_desc.detach(),
                              mu_desc_inf, logvar_desc_inf)

    # Reparamaterise
    u_corr = self.reparameterize(mu_corr, logvar_corr)
    u_desc = self.reparameterize(mu_desc, logvar_desc)
    u_corr_inf = self.reparameterize(mu_corr_inf, logvar_corr_inf)
    u_desc_inf = self.reparameterize(mu_desc_inf, logvar_desc_inf)

    # Decode from abduction
    x_desc_pred_logits, x_corr_pred_logits, y_pred_logits, \
      x_desc_cf, x_corr_cf, y_soc_cf_logits, _, y_full_cf_logits = self.decode(
        u_desc, u_corr, x_ind, s_bio, s_soc)
    
    # Decode from inference
    _, _, y_pred_inf_logits, *_ = self.decode(
        u_desc_inf, u_corr_inf, x_ind, s_bio, s_soc)

    # Reconstruction loss
    desc_recon_L = self.args.desc_a * self.reconstruction_loss(x_desc_pred_logits, x_desc, self.desc_meta)
    corr_recon_L = self.args.corr_a * self.reconstruction_loss(x_corr_pred_logits, x_corr, self.corr_meta)
    y_recon_L = self.args.pred_a * nn.BCEWithLogitsLoss()(y_pred_logits, y)

    recon_L = desc_recon_L + corr_recon_L + y_recon_L

    # KL loss
    kl_L = self.kl_loss(mu_corr, logvar_corr) + self.kl_loss(mu_desc, logvar_desc)
    

    # TC loss
    tc_L = self.tc_loss(u_desc, s_soc)

    # IECO fairness loss for a flipped Sociological sensitive attribute
    # We keep the Biological sensitive attribute constant

    # Counterfactual actual outcome (adbucted)
    y_soc_cf = torch.sigmoid(y_soc_cf_logits)

    # Second pass to infer the counterfactual prediction
    s_soc_flipped = 1 - s_soc

    mu_corr_cf, logvar_corr_cf, mu_desc_cf, logvar_desc_cf = self.encode(
        x_desc_cf, x_corr, x_ind, s_bio, y=None)
    
    u_corr_cf = self.reparameterize(mu_corr_cf, logvar_corr_cf)
    u_desc_cf = self.reparameterize(mu_desc_cf, logvar_desc_cf)

    _, _, y_pred_cf_logits, *_ = self.decode(
        u_desc_cf, u_corr_cf, x_ind, s_bio, s_soc_flipped)
    y_pred_cf_prob = torch.sigmoid(y_pred_cf_logits)

    y_pred_prob = torch.sigmoid(y_pred_inf_logits)
    
    fair_L = self.fair_loss(y, y_soc_cf, y_pred_prob, y_pred_cf_prob)

    # Latent Redundancy Loss
    u_redun_L = self.u_redundancy_loss(u_desc, u_corr, x_ind, s_soc, s_bio, y_pred_prob)

    # Total VAE obective
    # Fair Disentangled Negative ELBO = -M_ELBO + beta_tc * L_TC + beta_f * L_f
    total_vae_loss = recon_L + distill_weight*distill_L + kl_weight*kl_L + tc_weight*tc_L + self.args.fair_b*fair_L - 2*u_redun_L


    return total_vae_loss, desc_recon_L, corr_recon_L, y_recon_L, \
      kl_weight*kl_L, tc_weight*tc_L, self.args.fair_b*fair_L, distill_weight*distill_L, \
        y_pred_prob.detach(), \
           mu_desc.detach(), mu_corr.detach(), mu_desc_inf.detach(), mu_corr_inf.detach()

