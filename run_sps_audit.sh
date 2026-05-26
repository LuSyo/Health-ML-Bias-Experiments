pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/pathway_audit.py \
  --exp_name "24MAY_SPS" \
  --data "heart_disease_train.csv" \
  --mapping "configs/sps/features.json" \
  --early_stop_start 99 \
  --sps_iter 30 \
  --n_epochs 100 \
  --cross_val 5 \
  --desc_a 1.5 \
  --pred_a 2 \
  --tc_b 2 \
  --cf_invar_b 0.5 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --disc_step 1 \
  --kl_warm_up 20 \
  --tc_warm_up 10 \
  --cf_invar_warm_up 30 \
  --distill_warm_up 0 \
  --batch_size 32 \


