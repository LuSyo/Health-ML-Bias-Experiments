WITH weight_spine AS (
  SELECT 
    co.stay_id,
    COALESCE(
      arg_min(CASE WHEN ch.itemid IN (226512, 224639, 226846) AND ch.valuenum BETWEEN 20 AND 400 THEN ch.valuenum END, ch.charttime),
      80.0 -- Clinical default if no weight was recorded during the entire stay
    ) AS patient_weight
  FROM cohort co
  INNER JOIN chartevents ch ON co.stay_id = ch.stay_id
  WHERE ch.itemid IN (226512, 224639, 226846)
  GROUP BY co.stay_id
),

infusions_scaled AS (
  SELECT 
    ie.stay_id,
    ie.itemid,
    CASE 
      -- Dopamine Unit Correction
      -- Dopamine rate > 100 mcg/kg/min => unit error
      WHEN ie.itemid = 221662 AND ie.rate > 100.0 THEN 
        CASE 
          -- Mass-minute error, recorded as mcg/min
          WHEN (ie.rate / w.patient_weight) <= 50.0 THEN (ie.rate / w.patient_weight)
          -- Mass-hour error, recorded as mcg/hr
          WHEN (ie.rate / (w.patient_weight * 60.0)) <= 50.0 THEN (ie.rate / (w.patient_weight * 60.0))
          -- else clip at maximum ceiling 
          ELSE 50.0 
        END
      
      -- Norepinephrine / Epinephrine Unit Corrections
      WHEN ie.itemid IN (221906, 229617, 221289) AND ie.rate > 5.0 THEN 
        CASE 
          WHEN (ie.rate / w.patient_weight) <= 5.0 THEN (ie.rate / w.patient_weight)
          WHEN (ie.rate / (w.patient_weight * 60.0)) <= 5.0 THEN (ie.rate / (w.patient_weight * 60.0))
          ELSE 3.0 
        END
      
      ELSE ie.rate 
    END AS clean_rate
  FROM inputevents ie
  INNER JOIN weight_spine w ON ie.stay_id = w.stay_id
  WHERE ie.starttime <= ie.starttime + INTERVAL 1 DAY
)


SELECT 
  co.stay_id,

  -- 1. Mechanical Ventilation (Invasive or Non-Invasive Status)
  CASE WHEN EXISTS (
    SELECT 1 FROM procedureevents pe 
    WHERE pe.stay_id = co.stay_id 
      AND pe.itemid IN (225792, 225794, 224385)
      AND pe.starttime <= co.intime + INTERVAL 1 DAY
  ) THEN 1 ELSE 0 END AS vent_day1,
  
  -- 2. Continuous Renal Replacement Therapy (CRRT)
  CASE WHEN EXISTS (
    SELECT 1 FROM procedureevents pe 
    WHERE pe.stay_id = co.stay_id 
      AND pe.itemid IN (225441, 225802, 225803, 225805, 224149, 225809)
      AND pe.starttime <= co.intime + INTERVAL 1 DAY
  ) THEN 1 ELSE 0 END AS crrt_day1,
  
  -- 3. Antibiotics
  CASE WHEN EXISTS (
    SELECT 1 FROM prescriptions p 
    WHERE p.hadm_id = co.hadm_id 
      AND p.starttime <= co.intime + INTERVAL 1 DAY
  ) THEN 1 ELSE 0 END AS antibiotics_day1,

  -- 4. Continuous Max Vasopressor Dose Rates (Rescaled Layer)
  COALESCE(MAX(CASE WHEN inf.itemid = 221906 THEN inf.clean_rate END), 0.0) AS max_norepinephrine_rate,
  COALESCE(MAX(CASE WHEN inf.itemid IN (229617, 221289) THEN inf.clean_rate END), 0.0) AS max_epinephrine_rate,
  COALESCE(MAX(CASE WHEN inf.itemid = 221662 THEN inf.clean_rate END), 0.0) AS max_dopamine_rate

FROM cohort co
LEFT JOIN infusions_scaled inf ON co.stay_id = inf.stay_id
GROUP BY co.stay_id, co.intime, co.hadm_id;