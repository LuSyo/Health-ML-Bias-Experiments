python3 main.py \
  --exp_name "6-Mar_0" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_feature_mapping.json" \
  --n_epochs 100 \
  --vae_lr 0.001 \
  --disc_lr 0.0005 \
  --pred_a 5 \
  --tc_b 15 \
  --fair_b 1 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --ud_dim 3 \
  --uc_dim 3

