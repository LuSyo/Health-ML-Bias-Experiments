pkill -9 python
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

python3 src/bucket_audit.py \
  --exp_name "SBS_audit_2" \
  --data "heart_disease_cleaned.csv" \
  --mapping "configs/features.json" \

