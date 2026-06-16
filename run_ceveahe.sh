python3 src/main.py \
  --exp_name "0615_full_pipeline" \
  --training_data "heart_disease_train.csv" \
  --test_data "heart_disease_test.csv" \
  --mapping "configs/scm/uci_scm_config.json" \
  --n_epochs 400 \
  --early_stop_patience 15 \
  --early_stop_start 100 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --disc_step 1 \
  --desc_a 1 \
  --pred_a 1 \
  --tc_b 6 \
  --cf_invar_b 0.25 \
  --group_eta 0 \
  --kl_warm_up 10 \
  --tc_warm_up 30 \
  --cf_invar_warm_up 50 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --n_bootstraps 1000 \
  --target_recall 0.9 \
  --target_metric "recall" \
  --m_samples 5


