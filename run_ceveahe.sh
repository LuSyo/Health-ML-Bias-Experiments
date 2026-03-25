python3 src/main.py \
  --exp_name "SCM_0" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_0.json" \
  --n_epochs 200 \
  --vae_lr 0.001 \
  --disc_lr 0.0005 \
  --pred_a 2 \
  --tc_b 5 \
  --fair_b 1 \
  --u_ind_b 1 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1

