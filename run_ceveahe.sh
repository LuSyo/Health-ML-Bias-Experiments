python3 src/main.py \
  --exp_name "NEW_CEVAEHE_16MAY_6" \
  --training_data "heart_disease_train.csv" \
  --test_data "heart_disease_test.csv" \
  --mapping "configs/scm/uci_scm_config_simplified.json" \
  --n_epochs 200 \
  --vae_lr 0.0021 \
  --disc_lr 0.001 \
  --desc_a 1 \
  --pred_a 1 \
  --cf_invar_b 1 \
  --tc_b 0 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --cf_invar_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1


