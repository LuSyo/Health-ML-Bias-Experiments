WITH day1_vitals AS (
  SELECT 
    co.stay_id,

    -- Vitals Min/Max

    MIN(CASE WHEN ch.itemid = 220045 AND ch.valuenum BETWEEN 20 AND 250 THEN ch.valuenum END) AS hr_min,
    MAX(CASE WHEN ch.itemid = 220045 AND ch.valuenum BETWEEN 20 AND 250 THEN ch.valuenum END) AS hr_max,

    MIN(CASE WHEN ch.itemid = 220210 AND ch.valuenum BETWEEN 4 AND 80 THEN ch.valuenum END) AS rr_min,
    MAX(CASE WHEN ch.itemid = 220210 AND ch.valuenum BETWEEN 4 AND 80 THEN ch.valuenum END) AS rr_max,

    -- MIN(CASE WHEN ch.itemid IN (220179, 220059) THEN ch.valuenum END) AS sbp_min,
    -- MAX(CASE WHEN ch.itemid IN (220179, 220059) THEN ch.valuenum END) AS sbp_max,
    MIN(CASE WHEN ch.itemid IN (220179, 220059) AND ch.valuenum BETWEEN 40 AND 300 THEN ch.valuenum END) AS sbp_min,
    MAX(CASE WHEN ch.itemid IN (220179, 220059) AND ch.valuenum BETWEEN 40 AND 300 THEN ch.valuenum END) AS sbp_max,

    MIN(CASE WHEN ch.itemid IN (220180, 220060) AND ch.valuenum BETWEEN 20 AND 200 THEN ch.valuenum END) AS dbp_min,
    MAX(CASE WHEN ch.itemid IN (220180, 220060) AND ch.valuenum BETWEEN 20 AND 200 THEN ch.valuenum END) AS dbp_max,

    MIN(CASE WHEN ch.itemid IN (220052, 220181) AND ch.valuenum BETWEEN 30 AND 220 THEN ch.valuenum END) AS mbp_min,
    MAX(CASE WHEN ch.itemid IN (220052, 220181) AND ch.valuenum BETWEEN 30 AND 220 THEN ch.valuenum END) AS mbp_max,

    MIN(CASE 
          WHEN ch.itemid = 223762 AND ch.valuenum BETWEEN 25 AND 45 THEN ch.valuenum
          WHEN ch.itemid = 223761 AND ch.valuenum BETWEEN 77 AND 113 THEN (ch.valuenum - 32) * 5/9 
        END) AS temp_min,
    MAX(CASE 
          WHEN ch.itemid = 223762 AND ch.valuenum BETWEEN 25 AND 45 THEN ch.valuenum
          WHEN ch.itemid = 223761 AND ch.valuenum BETWEEN 77 AND 113 THEN (ch.valuenum - 32) * 5/9 
        END) AS temp_max,

    -- SpO2 (Min Only per Guan et al protocol)
    MIN(CASE WHEN ch.itemid = 220277 AND ch.valuenum BETWEEN 40 AND 100 THEN ch.valuenum END) AS spo2_min,

    -- SOFA factors
    MAX(CASE 
          WHEN ch.itemid = 223835 AND ch.valuenum BETWEEN 21 AND 100 THEN ch.valuenum
          WHEN ch.itemid = 223835 AND ch.valuenum BETWEEN 0.21 AND 1.0 THEN ch.valuenum * 100 
        END) AS fio2_max,
    MIN(CASE WHEN ch.itemid = 220739 THEN ch.valuenum END) AS gcs_eye_min,
    MIN(CASE WHEN ch.itemid = 223900 THEN ch.valuenum END) AS gcs_verbal_min,
    MIN(CASE WHEN ch.itemid = 223901 THEN ch.valuenum END) AS gcs_motor_min,

  FROM cohort co
  INNER JOIN chartevents ch ON co.stay_id = ch.stay_id
  WHERE ch.charttime BETWEEN co.intime AND co.intime + INTERVAL 1 DAY
  GROUP BY co.stay_id
),

global_weight AS (
  SELECT 
    co.stay_id,
    arg_min(
      CASE WHEN ch.itemid IN (226512, 224639, 226846) AND ch.valuenum BETWEEN 20 AND 400 THEN ch.valuenum END, 
      ch.charttime
    ) AS weight
  FROM cohort co
  INNER JOIN chartevents ch ON co.stay_id = ch.stay_id
  WHERE ch.itemid IN (226512, 224639, 226846)
  GROUP BY co.stay_id
)

SELECT 
  v.*,
  w.weight
FROM day1_vitals v
LEFT JOIN global_weight w ON v.stay_id = w.stay_id;