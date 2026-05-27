python3 src/main.py \
  --exp_name "27_MAY_1" \
  --training_data "heart_disease_train.csv" \
  --test_data "heart_disease_test.csv" \
  --mapping "configs/scm/uci_scm_config_simplified.json" \
  --n_epochs 200 \
  --early_stop_patience 15 \
  --early_stop_start 100 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --desc_a 1.25 \
  --pred_a 4 \
  --cf_invar_b 1 \
  --tc_b 6 \
  --group_eta 0 \
  --kl_warm_up 20 \
  --tc_warm_up 0 \
  --cf_invar_warm_up 30 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1 


