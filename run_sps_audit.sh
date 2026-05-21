pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/pathway_audit.py \
  --exp_name "21MAY_SPS_test" \
  --data "heart_disease_train.csv" \
  --mapping "configs/sps/features.json" \
  --sps_iter 1 \
  --n_epochs 100 \
  --cross_val 1 \
  --desc_a 1 \
  --pred_a 1 \
  --tc_b 1 \
  --cf_invar_b 1 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --disc_step 1 \
  --kl_warm_up 20 \
  --tc_warm_up 10 \
  --cf_invar_warm_up 50 \
  --distill_warm_up 0 \
  --batch_size 32 \


