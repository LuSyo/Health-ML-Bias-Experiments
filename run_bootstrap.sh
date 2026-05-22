pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/bootstrap.py \
  --exp_name "21MAY_CLASSIFIERS_B_1_train_set" \
  --data "heart_disease_train.csv" \
  --mapping "configs/scm/uci_scm_config_simplified_3.json" \
  --cevaehe "21MAY_CEVAEHE_B_1_cevaehe.pth" \
  --cf_dataset "21MAY_CEVAEHE_B_1/test_counterfactuals.csv" \
  --latent_dataset "21MAY_CEVAEHE_B_1/test_latent_space.csv" \
  --n_runs 50

