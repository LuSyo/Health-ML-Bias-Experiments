pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/bootstrap.py \
  --exp_name "SCM_79_bootstrap" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_79.json" \
  --cevaehe "SCM_79_cevaehe.pth" \
  --cf_dataset "SCM_79/counterfactuals.csv" \
  --latent_dataset "SCM_79/latent_spaces.csv" \
  --n_epochs 200 \
  --vae_lr 0.001 \
  --disc_lr 0.0001 \
  --pred_a 2 \
  --tc_b 8 \
  --fair_b 2 \
  --u_ind_b 1 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --n_runs 50

