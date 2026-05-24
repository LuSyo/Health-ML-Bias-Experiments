python3 src/finetuning.py \
  --exp_name "21MAY_finetuning_B" \
  --data "heart_disease_train.csv" \
  --mapping "configs/scm/uci_scm_config_simplified_3.json" \
  --param_space "configs/finetuning/param_space.json" \
  --early_stop_patience 10 \
  --early_stop_start 100 \
  --param_iter 36 \
  --cross_val 4 \
  --n_epochs 200 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --kl_warm_up 20 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --cf_invar_warm_up 30 \
  --batch_size 32 \
  --disc_step 1

