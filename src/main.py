import pandas as pd
import numpy as np
import os
import gc

from config import Config
from utils import parse_args, load_feature_mapping, set_global_seeds, setup_logger
from cevaehe.data_loader import make_bucketed_loader
from cevaehe.model import CEVAEHE
from cevaehe.train import train_cevaehe
from cevaehe.test import generate_fair_dataset, test_ceveahe
from plots import train_val_loss_curve, disc_tc_loss_curve, all_VAE_losses_curve, training_accuracy_curve, u_clustering_analysis, grad_curve
from metrics import get_cca

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
  logger.info(f'Hidden layers dimension: {args.h_dim}')
  logger.info(f'Activation function: {args.act_fn}')
  logger.info(f'Corr. recon. loss alpha: {args.corr_a}')
  logger.info(f'Desc. recon. loss alpha: {args.desc_a}')
  logger.info(f'Prediction loss alpha: {args.pred_a}')
  logger.info(f'Fair loss beta: {args.fair_b}')
  logger.info(f'TC loss beta: {args.tc_b}')

  logger.info('='*30)

  try:
    # Load the dataset
    dataset = pd.read_csv(Config.DATA_DIR + args.data)

    # Load the feature mapping
    feature_mapping = load_feature_mapping(args.mapping)

    # Data loaders
    train_loader, val_loader, test_loader = make_bucketed_loader(dataset, feature_mapping,
                                                                 test_size=0.2,
                                                                 batch_size=args.batch_size, seed=args.seed)

    # Feature metadata
    ind_meta = feature_mapping['ind']
    desc_meta = feature_mapping['desc']
    corr_meta = feature_mapping['corr']
    sens_meta = feature_mapping['sens']

    model = CEVAEHE(ind_meta, desc_meta, corr_meta, sens_meta, 
                  args=args)
    
    
    logger.info(f'U_corr dimension: {model.uc_dim}')
    logger.info(f'U_desc dimension: {model.ud_dim}')
    
    training_log, train_results = train_cevaehe(
      model,
      train_loader,
      val_loader,
      logger,
      args
    )

    logger.info(f'END EXPERIMENT {args.exp_name}')  

    training_metrics = pd.DataFrame(training_log)

    train_val_loss_fig = train_val_loss_curve(training_metrics)
    train_val_loss_fig.savefig(f'{results_path}/train_val_loss_curve.png', bbox_inches='tight')
    disc_tc_loss_fig = disc_tc_loss_curve(training_metrics)
    disc_tc_loss_fig.savefig(f'{results_path}/disc_tc_loss_curve.png', bbox_inches='tight')
    all_VAE_losses_fig = all_VAE_losses_curve(training_metrics)
    all_VAE_losses_fig.savefig(f'{results_path}/VAE_losses_curve.png', bbox_inches='tight')
    training_accuracy_fig = training_accuracy_curve(training_metrics)
    training_accuracy_fig.savefig(f'{results_path}/training_accuracy_curve.png', bbox_inches='tight')
    train_u_clustering_analysis_fig = u_clustering_analysis(train_results, mode="training")
    train_u_clustering_analysis_fig.savefig(f'{results_path}/train_u_clustering_analysis.png', bbox_inches='tight')

    # grad_norm_fig = grad_curve(training_metrics)
    # grad_norm_fig.savefig(f'{results_path}/grad_norm_curve.png', bbox_inches='tight')

    test_results, perf_metrics, strat_perf_metrics = test_ceveahe(model, test_loader, logger, args)

    test_u_clustering_analysis_fig = u_clustering_analysis(test_results)
    test_u_clustering_analysis_fig.savefig(f'{results_path}/test_u_clustering_analysis.png', bbox_inches='tight')


    # save stratified perf metrics as a markdown table
    strat_perf_summary = pd.DataFrame({
        'group': [0, 1],
        'accuracy': [strat_perf_metrics['accuracy_0'], strat_perf_metrics['accuracy_1']],
        'roc_auc': [strat_perf_metrics['roc_auc_0'], strat_perf_metrics['roc_auc_1']],
        'fnr': [strat_perf_metrics['fnr_0'], strat_perf_metrics['fnr_1']],
        'fpr': [strat_perf_metrics['fpr_0'], strat_perf_metrics['fpr_1']],
        'brier_score': [strat_perf_metrics['brier_score_0'], strat_perf_metrics['brier_score_1']],
    })
    strat_perf_summary.to_markdown(f'{results_path}/stratified_perf_metrics.txt', index=False)

    # CAUSAL MODEL VALIDATION
    # Independence of u_desc and u_corr
    u_desc_matrix = np.stack(test_results['u_desc'].values)
    u_corr_matrix = np.stack(test_results['u_corr'].values)
    u_u_cca = get_cca(u_desc_matrix, u_corr_matrix)
    logger.info(f'Canonical Correlation Analysis between U_corr and U_desc: {u_u_cca:.4f}')

    # Generate counterfactual and latent space datasets (fair dataset)
    logger.info('Saving latent and counterfactual datasets...')
    datasets_path = f'{Config.DATA_DIR}/{args.exp_name}'
    os.makedirs(datasets_path, exist_ok=True)
    counterfactuals_df, latent_spaces_df = generate_fair_dataset(model, dataset, feature_mapping, args)
    counterfactuals_df.to_csv(f'{datasets_path}/counterfactuals.csv', index=False)
    latent_spaces_df.to_csv(f'{datasets_path}/latent_spaces.csv', index=False)

    del counterfactuals_df, latent_spaces_df

    gc.collect()

  except Exception as e:
    logger.error(f'Experiment failed: {str(e)}', exc_info=True)
  
if __name__ == "__main__":
  main()
  