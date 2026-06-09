WITH cohort_diagnoses AS (
  SELECT 
    co.hadm_id,
    REPLACE(diag.icd_code, '.', '') AS clean_code,
    diag.icd_version
  FROM cohort co
  INNER JOIN diagnoses_icd diag 
    ON co.hadm_id = diag.hadm_id
)
SELECT 
  hadm_id,

  -- 1. Hypertension
  MAX(CASE WHEN (icd_version = 10 
      AND clean_code IN ('I10', 'I11', 'I12', 'I13', 'I15', 'N262'))
    OR (icd_version = 9 
      AND clean_code IN ('4011', '4019', '40210', '40290', '40410', '40490', '40511', '40519', '40591', '40599')) 
    THEN 1 ELSE 0 END) AS comorb_hypertension,

  -- 2. Diabetes Mellitus
  MAX(CASE WHEN (icd_version = 10 
      AND LEFT(clean_code, 3) IN ('E10', 'E11', 'E12', 'E13', 'E14')
    OR (icd_version = 9 
      AND clean_code LIKE '250%'))
    THEN 1 ELSE 0 END) AS comorb_diabetes,
  
  -- 3. Chronic Kidney Disease (CKD): Anchored to N18 and 585
  MAX(CASE WHEN (icd_version = 10 
      AND clean_code LIKE 'N18%') 
    OR (icd_version = 9 
      AND clean_code LIKE '585%') 
    THEN 1 ELSE 0 END) AS comorb_ckd,

  -- 4. Old Myocardial Infarction (MI)
  MAX(CASE WHEN (icd_version = 10 
      AND clean_code = 'I252') 
    OR (icd_version = 9 
      AND clean_code = '412')
    THEN 1 ELSE 0 END) AS comorb_mi,
              
  -- 5. Chronic Lung Disease
  MAX(CASE WHEN (icd_version = 10 
      AND LEFT(clean_code, 3) IN ('J41', 'J42', 'J43', 'J44')) 
    OR (icd_version = 9 
      AND LEFT(clean_code, 3) IN ('491', '492', '496'))
    THEN 1 ELSE 0 END) AS comorb_lung,
              
  -- 6. Chronic Liver Disease
  MAX(CASE WHEN (icd_version = 10 
      AND (clean_code LIKE 'K70%' OR clean_code LIKE 'K74%')) 
    OR (icd_version = 9 
      AND (clean_code LIKE '571%' OR clean_code LIKE '572%')) 
    THEN 1 ELSE 0 END) AS comorb_liver,

  -- 7. Alcohol use
  MAX(CASE WHEN (icd_version = 10 
      AND (clean_code IN ('V113', 'E52', 'G621', 'I426', 'K292', 'T51', 'Z714', 'Z658') 
        OR LEFT(clean_code, 3) IN ('F10', 'K70'))) 
    OR (icd_version = 9 
      AND (clean_code IN ('30393', '30503') OR LEFT(clean_code, 3) = '291')) 
    THEN 1 ELSE 0 END) AS comorb_alcohol

FROM cohort_diagnoses
GROUP BY hadm_id;