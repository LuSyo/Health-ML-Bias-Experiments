pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/bootstrap.py \
  --exp_name "17-Mar_9_bootstrap" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_1.json" \
  --cevaehe "17-Mar_9_cevaehe.pth" \
  --cf_dataset "17-Mar_9/counterfactuals.csv" \
  --latent_dataset "17-Mar_9/latent_spaces.csv" \
  --n_epochs 200 \
  --vae_lr 0.001 \
  --disc_lr 0.0001 \
  --pred_a 2 \
  --tc_b 5 \
  --fair_b 1 \
  --u_ind_b 1 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --n_runs 50

