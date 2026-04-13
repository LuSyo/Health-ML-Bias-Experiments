python3 src/main.py \
  --exp_name "params_config_0" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_79.json" \
  --n_epochs 200 \
  --vae_lr 0.0015 \
  --disc_lr 0.0005 \
  --corr_a 1 \
  --desc_a 1.25 \
  --pred_a 1 \
  --fair_b 4 \
  --tc_b 6 \
  --u_ind_b 1.5 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1

