python3 src/main.py \
  --exp_name "STEP_A_EXP_NAME" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_config.json" \
  --n_epochs 200 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --corr_a 1 \
  --desc_a 1 \
  --pred_a 1 \
  --fair_b 2 \
  --tc_b 4 \
  --u_ind_b 0.5 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1

