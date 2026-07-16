pkill -9 python
sync
# sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/pathway_audit.py \
  --exp_name "prev40_dim4_cont_SPS_norm" \
  --data "synth/prev40_dim4_cont/train.csv" \
  --mapping "configs/sps/toy_features_4.json" \
  --sps_mode "normal" \
  --sps_iter 24 \
  --cross_val 4 \
  --n_epochs 300 \
  --early_stop_start 50 \
  --vae_lr 0.0021 \
  --disc_lr 0.0007 \
  --disc_step 1 \
  --desc_a 1\
  --pred_a 1.5 \
  --cf_invar_b 0.75 \
  --tc_b 12 \
  --group_eta 0 \
  --kl_warm_up 10 \
  --tc_warm_up 20 \
  --cf_invar_warm_up 30 \
  --distill_warm_up 0 \
  --batch_size 512 \


