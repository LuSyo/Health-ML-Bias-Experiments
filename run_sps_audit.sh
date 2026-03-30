pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/pathway_audit.py \
  --exp_name "SPS_audit_100_with_ind" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/features_with_ind.json" \
  --sps_iter 100 \
  --sps_epochs 100 \
  --pred_a 2 \
  --tc_b 8 \
  --fair_b 2 \
  --vae_lr 0.0015 \
  --disc_lr 0.0005 \
  --disc_step 1 \
  --u_ind_b 1 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \

