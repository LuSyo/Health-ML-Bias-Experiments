python3 src/finetuning.py \
  --exp_name "21MAY_finetuning_B" \
  --data "heart_disease_train.csv" \
  --mapping "configs/scm/uci_scm_config_simplified_3.json" \
  --param_space "configs/finetuning/param_space.json" \
  --n_epochs 200 \
  --early_stop_patience 10 \
  --early_stop_start 199 \
  --param_iter 20 \
  --cross_val 4 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --disc_step 1 \
  --kl_warm_up 10 \
  --tc_warm_up 30 \
  --cf_invar_warm_up 50 \
  --distill_warm_up 0 \
  --batch_size 128

