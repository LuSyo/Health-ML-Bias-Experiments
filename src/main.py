import pandas as pd
import numpy as np
import os
import gc

from config import Config
from utils import parse_args, load_config, set_global_seeds, setup_logger
from cevaehe_new.data_loader import make_bucketed_loader
from cevaehe_new.model import CEVAEHE
from cevaehe_new.train import train_cevaehe
from cevaehe_new.test import test_ceveahe, generate_fair_dataset
from plots import train_val_recon_loss_curve, disc_tc_loss_curve, all_VAE_losses_curve, u_clustering_analysis, disc_acc_train_val_curve, grad_curve

def main():
  args = parse_args()

  set_global_seeds(args.seed)

  results_path = f'{args.root_dir}{Config.RESULTS_DIR}{args.exp_name}'
  os.makedirs(results_path, exist_ok=True)

  # Initialise logger
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  logger.info(f'EXPERIMENT START: {args.exp_name}')
  logger.info(f'Device {args.device}')
  logger.info(f'Data: {args.data}')
  logger.info(f'Mapping: {args.mapping}')
  logger.info(f'Batch size: {args.batch_size}')
  logger.info(f'Epochs: {args.n_epochs}')
  logger.info(f'VAE learning rate: {args.vae_lr}')
  logger.info(f'Discriminator learning rate: {args.disc_lr}')
  logger.info(f'Discriminator learning period: {args.disc_step}')
  logger.info(f'Distillation Warm-up: {args.distill_warm_up}')
  logger.info(f'TC Loss Warm-up: {args.tc_warm_up}')
  logger.info(f'KL Warm-up: {args.kl_warm_up}')
  logger.info(f'CF Invariance Warm-up: {args.cf_invar_warm_up}')
  logger.info(f'Hidden layers dimension: {args.h_dim}')
  logger.info(f'Activation function: {args.act_fn}')
  logger.info(f'Desc. recon. loss alpha: {args.desc_a}')
  logger.info(f'Prediction loss alpha: {args.pred_a}')
  logger.info(f'TC loss beta: {args.tc_b}')
  logger.info(f'Latent CF Invariance loss beta: {args.cf_invar_b}')

  try:
    # Load the datasets
    training_dataset = pd.read_csv(Config.DATA_DIR + args.training_data)
    test_dataset = pd.read_csv(Config.DATA_DIR + args.test_data)

    # Load the feature mapping
    feature_mapping = load_config(args.mapping)

    # TRAINING Data loaders
    train_loader, val_loader = make_bucketed_loader(
      training_dataset, 
      feature_mapping,
      val_size=0.2,
      batch_size=args.batch_size,
      seed=args.seed)

    # Feature metadata
    indcorr_meta = feature_mapping['indcorr']
    desc_meta = feature_mapping['desc']
    sens_meta = feature_mapping['sens']

    model = CEVAEHE(desc_meta, sens_meta, args=args)
    
    logger.info(f'U_desc dimension: {model.ud_dim}')
    logger.info('='*30)
    
    training_log, train_results = train_cevaehe(
      model,
      train_loader,
      val_loader,
      logger,
      args
    )

    logger.info(f'END EXPERIMENT {args.exp_name}')  

    training_metrics = pd.DataFrame(training_log)

    train_val_recon_loss_fig = train_val_recon_loss_curve(training_metrics, sens_groups=model.sens_groups)
    train_val_recon_loss_fig.savefig(f'{results_path}/train_val_recon_loss_curve.png', bbox_inches='tight')

    train_val_disc_acc_fig = disc_acc_train_val_curve(training_metrics)
    train_val_disc_acc_fig.savefig(f'{results_path}/train_val_disc_acc_fig.png', bbox_inches='tight')

    disc_tc_loss_fig = disc_tc_loss_curve(training_metrics)
    disc_tc_loss_fig.savefig(f'{results_path}/disc_tc_loss_curve.png', bbox_inches='tight')

    all_VAE_losses_fig = all_VAE_losses_curve(training_metrics)
    all_VAE_losses_fig.savefig(f'{results_path}/VAE_losses_curve.png', bbox_inches='tight')

    train_u_clustering_analysis_fig = u_clustering_analysis(train_results, mode="training")
    train_u_clustering_analysis_fig.savefig(f'{results_path}/train_u_clustering_analysis.png', bbox_inches='tight')

    grad_curves_fig = grad_curve(training_metrics)
    grad_curves_fig.savefig(f'{results_path}/grad_curves.png', bbox_inches='tight')

    if train_results is not None:      
      mu_desc_0 = np.stack(train_results.loc[train_results['x_sens'] == 0, 'mu_desc'].tolist())
      logvar_desc_0 = np.stack(train_results.loc[train_results['x_sens'] == 0, 'mu_desc'].tolist())
      mu_desc_1 = np.stack(train_results.loc[train_results['x_sens'] == 1, 'mu_desc'].tolist())
      logvar_desc_1 = np.stack(train_results.loc[train_results['x_sens'] == 1, 'mu_desc'].tolist())

      latent_dist_dict = {}
      for dim in range(model.ud_dim):
        u_desc_mu_0 = mu_desc_0[dim].mean()
        logvars_0 = logvar_desc_0[dim]
        u_desc_var_0 = np.exp(logvars_0).mean()

        u_desc_mu_1 = mu_desc_1[dim].mean()
        logvars_1 = logvar_desc_1[dim]
        u_desc_var_1 = np.exp(logvars_1).mean()

        latent_dist_dict[f"Udesc_{dim} Mean"] = [u_desc_mu_0, u_desc_mu_1]
        latent_dist_dict[f"Udesc_{dim} Variance"] = [u_desc_var_0, u_desc_var_1]

      strat_latent_dist_params = pd.DataFrame({
        "S Group": [0, 1]
      } | latent_dist_dict)

      strat_latent_dist_params.to_markdown(f'{results_path}/stratified_latent_dist_params.txt', index=False)

    ### UNSEEN DATA: TEST RESULTS AND LATENT/CF GENERATION
    set_global_seeds(args.seed)
    
    # TEST Data loader
    test_loader, _ = make_bucketed_loader(
      test_dataset, 
      feature_mapping,
      val_size=0,
      batch_size=args.batch_size,
      seed=args.seed)

    test_outputs, _= test_ceveahe(model, test_loader, logger, args)

    test_u_clustering_analysis_fig = u_clustering_analysis(test_outputs)
    test_u_clustering_analysis_fig.savefig(f'{results_path}/test_u_clustering_analysis.png', bbox_inches='tight')

    # Generate counterfactual and latent space datasets (fair dataset)
    logger.info('Saving latent and counterfactual datasets...')
    datasets_path = f'{Config.DATA_DIR}/{args.exp_name}'
    os.makedirs(datasets_path, exist_ok=True)

    test_counterfactuals_df, test_latent_spaces_df = generate_fair_dataset(model, test_dataset, feature_mapping, args)
    test_counterfactuals_df.to_csv(f'{datasets_path}/test_counterfactuals.csv', index=False)
    test_latent_spaces_df.to_csv(f'{datasets_path}/test_latent_space.csv', index=False)

    train_counterfactuals_df, train_latent_spaces_df = generate_fair_dataset(model, training_dataset, feature_mapping, args)
    train_counterfactuals_df.to_csv(f'{datasets_path}/train_counterfactuals.csv', index=False)
    train_latent_spaces_df.to_csv(f'{datasets_path}/train_latent_space.csv', index=False)

    del test_counterfactuals_df, test_latent_spaces_df
    del train_counterfactuals_df, train_latent_spaces_df
    del model

    gc.collect()

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)
  
if __name__ == "__main__":
  main()
  