pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/bootstrap.py \
  --exp_name "SCM2_15Apr_bootstrap" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_2.json" \
  --cevaehe "SCM2_15Apr_cevaehe.pth" \
  --cf_dataset "SCM2_15Apr/counterfactuals.csv" \
  --latent_dataset "SCM2_15Apr/latent_spaces.csv" \
  --n_epochs 200 \
  --vae_lr 0.0015 \
  --disc_lr 0.0005 \
  --corr_a 1.5 \
  --desc_a 1.25 \
  --pred_a 1 \
  --fair_b 2 \
  --tc_b 8 \
  --u_ind_b 0.5 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1 \
  --n_runs 50

