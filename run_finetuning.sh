python3 src/finetuning.py \
  --exp_name "finetuning_recon" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_config.json" \
  --param_space "configs/recon_param_space.json" \
  --param_iter 15 \
  --cross_val 4 \
  --n_epochs 100 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1

