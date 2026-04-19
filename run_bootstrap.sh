pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/bootstrap.py \
  --exp_name "BOOTSTRAP" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_config.json" \
  --cevaehe "STEP_A_EXP_NAME_cevaehe.pth" \
  --cf_dataset "STEP_A_EXP_NAME/counterfactuals.csv" \
  --latent_dataset "STEP_A_EXP_NAME/latent_spaces.csv" \
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
  --disc_step 1 \
  --n_runs 50

