pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/pathway_audit.py \
  --exp_name "SPS_14Apr" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/features.json" \
  --sps_iter 50 \
  --sps_epochs 100 \
  --pred_a 1 \
  --tc_b 6 \
  --fair_b 4 \
  --vae_lr 0.0015 \
  --disc_lr 0.0005 \
  --disc_step 1 \
  --u_ind_b 1 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \

