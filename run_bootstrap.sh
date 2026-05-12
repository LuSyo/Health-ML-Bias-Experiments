pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/bootstrap.py \
  --exp_name "12MAY_SYNTH_classifiers_fair" \
  --data "synth/11MAY/simple_fair_test.csv" \
  --mapping "configs/scm/synth_simple_scm_fair.json" \
  --cevaehe "12MAY_SYNTH_CEVAE_fair_cevaehe.pth" \
  --cf_dataset "12MAY_SYNTH_CEVAE_fair/counterfactuals.csv" \
  --latent_dataset "12MAY_SYNTH_CEVAE_fair/latent_spaces.csv" \
  --n_runs 50 \
  --target_ppv 0.9

