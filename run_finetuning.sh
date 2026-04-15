python3 src/finetuning.py \
  --exp_name "finetuning_15Apr_2" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/uci_scm_2.json" \
  --param_space "configs/disentang_params_space_1.json" \
  --param_iter 20 \
  --cross_val 4 \
  --n_epochs 200 \
  --vae_lr 0.0015 \
  --disc_lr 0.0005 \
  --kl_warm_up 0 \
  --tc_warm_up 0 \
  --distill_warm_up 0 \
  --batch_size 32 \
  --disc_step 1

