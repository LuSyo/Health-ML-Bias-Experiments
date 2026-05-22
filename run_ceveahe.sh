python3 src/main.py \
  --exp_name "21MAY_CEVAEHE_B_1" \
  --training_data "heart_disease_train.csv" \
  --test_data "heart_disease_test.csv" \
  --mapping "configs/scm/uci_scm_config_simplified_3.json" \
  --n_epochs 200 \
  --early_stop_patience 15 \
  --early_stop_start 80 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --desc_a 1.25 \
  --pred_a 4 \
  --cf_invar_b 1 \
  --tc_b 6 \
  --kl_warm_up 20 \
  --tc_warm_up 10 \
  --cf_invar_warm_up 50 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1 


