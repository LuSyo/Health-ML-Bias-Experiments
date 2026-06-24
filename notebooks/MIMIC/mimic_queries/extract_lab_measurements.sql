SELECT 
  co.stay_id,

  MIN(CASE WHEN le.itemid = 51222 AND le.valuenum BETWEEN 2 AND 25 THEN le.valuenum END) AS hemoglobin_min,
  MAX(CASE WHEN le.itemid = 51222 AND le.valuenum BETWEEN 2 AND 25 THEN le.valuenum END) AS hemoglobin_max,

  MIN(CASE WHEN le.itemid IN (51300, 51301) AND le.valuenum BETWEEN 0.1 AND 300 THEN le.valuenum END) AS wbc_min,
  MAX(CASE WHEN le.itemid IN (51300, 51301) AND le.valuenum BETWEEN 0.1 AND 300 THEN le.valuenum END) AS wbc_max,
  
  MIN(CASE WHEN le.itemid = 51265 AND le.valuenum BETWEEN 5 AND 1500 THEN le.valuenum END) AS platelets_min,
  MAX(CASE WHEN le.itemid = 51265 AND le.valuenum BETWEEN 5 AND 1500 THEN le.valuenum END) AS platelets_max,
  
  MIN(CASE WHEN le.itemid = 51006 AND le.valuenum BETWEEN 1 AND 300 THEN le.valuenum END) AS bun_min,
  MAX(CASE WHEN le.itemid = 51006 AND le.valuenum BETWEEN 1 AND 300 THEN le.valuenum END) AS bun_max,
  
  MIN(CASE WHEN le.itemid = 50912 AND le.valuenum BETWEEN 0.1 AND 25 THEN le.valuenum END) AS creatinine_min,
  MAX(CASE WHEN le.itemid = 50912 AND le.valuenum BETWEEN 0.1 AND 25 THEN le.valuenum END) AS creatinine_max,
  
  MIN(CASE WHEN le.itemid IN (50809, 50931) AND le.valuenum BETWEEN 10 AND 800 THEN le.valuenum END) AS glucose_min,
  MAX(CASE WHEN le.itemid IN (50809, 50931) AND le.valuenum BETWEEN 10 AND 800 THEN le.valuenum END) AS glucose_max,
  
  MIN(CASE WHEN le.itemid = 50868 AND le.valuenum BETWEEN 1 AND 50 THEN le.valuenum END) AS aniongap_min,
  MAX(CASE WHEN le.itemid = 50868 AND le.valuenum BETWEEN 1 AND 50 THEN le.valuenum END) AS aniongap_max,
  
  MIN(CASE WHEN le.itemid IN (50971, 50822) AND le.valuenum BETWEEN 1.5 AND 10 THEN le.valuenum END) AS potassium_min,
  MAX(CASE WHEN le.itemid IN (50971, 50822) AND le.valuenum BETWEEN 1.5 AND 10 THEN le.valuenum END) AS potassium_max,
  
  MIN(CASE WHEN le.itemid IN (50983, 50824) AND le.valuenum BETWEEN 90 AND 175 THEN le.valuenum END) AS sodium_min,
  MAX(CASE WHEN le.itemid IN (50983, 50824) AND le.valuenum BETWEEN 90 AND 175 THEN le.valuenum END) AS sodium_max,
  
  MIN(CASE WHEN le.itemid = 50893 AND le.valuenum BETWEEN 2 AND 20 THEN le.valuenum END) AS calcium_min,
  MAX(CASE WHEN le.itemid = 50893 AND le.valuenum BETWEEN 2 AND 20 THEN le.valuenum END) AS calcium_max,
  
  MIN(CASE WHEN le.itemid = 50910 AND le.valuenum BETWEEN 1 AND 300000 THEN le.valuenum END) AS ck_cpk_min,
  MAX(CASE WHEN le.itemid = 50910 AND le.valuenum BETWEEN 1 AND 300000 THEN le.valuenum END) AS ck_cpk_max,
  
  MIN(CASE WHEN le.itemid = 50911 AND le.valuenum BETWEEN 1 AND 50000 THEN le.valuenum END) AS ck_mb_min,
  MAX(CASE WHEN le.itemid = 50911 AND le.valuenum BETWEEN 1 AND 50000 THEN le.valuenum END) AS ck_mb_max,
  
  MIN(CASE WHEN le.itemid = 50963 AND le.valuenum BETWEEN 1 AND 100000 THEN le.valuenum END) AS ntbnp_min,
  MAX(CASE WHEN le.itemid = 50963 AND le.valuenum BETWEEN 1 AND 100000 THEN le.valuenum END) AS ntbnp_max,

  MIN(CASE WHEN le.itemid = 50821 AND le.valuenum BETWEEN 20 AND 700 THEN le.valuenum END) AS pao2_min,  

  MAX(CASE WHEN le.itemid = 50885 AND le.valuenum BETWEEN 0.1 AND 50 THEN le.valuenum END) AS bilirubin_max
FROM cohort co
INNER JOIN 
  labevents le 
  ON co.hadm_id = le.hadm_id
WHERE le.charttime BETWEEN co.intime AND co.intime + INTERVAL 1 DAY
GROUP BY co.stay_id;