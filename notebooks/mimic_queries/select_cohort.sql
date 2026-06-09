WITH first_icu_stays AS (
  SELECT 
    subject_id,
    hadm_id,
    stay_id,
    intime,
    los
  FROM icustays
  QUALIFY ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime ASC) = 1
),

-- BASE FILTER: First ICU stays for patients over 18 that lasted more than 2 days

eligible_adults AS (
  SELECT 
    icu.subject_id,
    icu.hadm_id,
    icu.stay_id,
    icu.intime,
    icu.los,
    adm.race,
    adm.admittime,
    pat.gender,
    (pat.anchor_age + (EXTRACT(YEAR FROM icu.intime) - pat.anchor_year)) AS admission_age
  FROM 
    first_icu_stays icu
  INNER JOIN 
    admissions adm 
    ON icu.hadm_id = adm.hadm_id
  INNER JOIN 
    patients pat 
    ON icu.subject_id = pat.subject_id
  WHERE 
    icu.los > 2.0
    AND (pat.anchor_age + (EXTRACT(YEAR FROM icu.intime) - pat.anchor_year)) >= 18
),

-- EXCLUSION 1: AF billing code in ANY prior admission

historical_af AS (
  SELECT DISTINCT ea.subject_id
  FROM eligible_adults ea
  INNER JOIN 
    diagnoses_icd diag 
    ON ea.subject_id = diag.subject_id
  INNER JOIN 
    admissions adm_past 
    ON diag.hadm_id = adm_past.hadm_id
  WHERE 
    adm_past.admittime < ea.admittime 
    AND (diag.icd_code = '42731' OR diag.icd_code LIKE 'I48%')
),

-- EXCLUSION 2: Early-onset / Pre-existing presentation inside ICU Day 1 window

icu_day1_af AS (
  SELECT DISTINCT ea.stay_id
  FROM eligible_adults ea
  INNER JOIN 
    chartevents chart 
    ON ea.stay_id = chart.stay_id
  WHERE 
    chart.itemid = 220048
    AND chart.charttime BETWEEN ea.intime AND ea.intime + INTERVAL 1 DAY
    AND (LOWER(chart.value) LIKE '%atrial fibrillation%' OR LOWER(chart.value) LIKE '%a-fib%')
),

-- CASE PRE-INCLUSION: New Onset AF after ICU stay Day 1

icu_day2_plus_af AS (
  SELECT DISTINCT ea.stay_id
  FROM eligible_adults ea
  INNER JOIN 
    chartevents chart 
    ON ea.stay_id = chart.stay_id
  WHERE 
    chart.itemid = 220048
    AND chart.charttime > ea.intime + INTERVAL 1 DAY 
    AND (LOWER(chart.value) LIKE '%atrial fibrillation%' OR LOWER(chart.value) LIKE '%a-fib%')
),

-- Identify any AF billing code assigned to the CURRENT hospital admission
current_adm_af_icd AS (
  SELECT DISTINCT ea.hadm_id
  FROM eligible_adults ea
  INNER JOIN 
    diagnoses_icd diag 
    ON ea.hadm_id = diag.hadm_id
  WHERE 
    (diag.icd_code = '42731' OR diag.icd_code LIKE 'I48%')
),

-- EXCLUSION 3: History of cardiac surgery, including valve surgery and coronary artery bypass grafting

cardiac_surgery AS (
  SELECT DISTINCT ea.subject_id
  FROM eligible_adults ea
  INNER JOIN 
    procedures_icd proc 
    ON ea.subject_id = proc.subject_id
  INNER JOIN 
    admissions adm_proc 
    ON proc.hadm_id = adm_proc.hadm_id
  WHERE 
    adm_proc.admittime <= ea.admittime
    AND (
      -- ICD-9: 361x (CABG), 351x/352x (Valves)
      proc.icd_code LIKE '361%' 
      OR proc.icd_code LIKE '351%' 
      OR proc.icd_code LIKE '352%'
      -- ICD-10-PCS: 021% (Bypass Heart), 02R% (Replace Heart Valve)
      OR proc.icd_code LIKE '021%' 
      OR proc.icd_code LIKE '02R%'
    )
)

-- FINAL SELECTION
SELECT 
  ea.*,
  CASE WHEN t.stay_id IS NOT NULL THEN 1 ELSE 0 END AS target_noaf
FROM eligible_adults ea
LEFT JOIN 
  icu_day2_plus_af t 
  ON ea.stay_id = t.stay_id
WHERE 
  ea.subject_id NOT IN (SELECT subject_id FROM historical_af)       -- Exclude pre existing history
  AND ea.stay_id NOT IN (SELECT stay_id FROM icu_day1_af)           -- Exclude early onset events
  AND ea.subject_id NOT IN (SELECT subject_id FROM cardiac_surgery) -- Exclude cardiac surgery history
  AND (
    (t.stay_id IS NOT NULL AND ea.hadm_id IN (SELECT hadm_id FROM current_adm_af_icd))    -- TRUE CASE: bedside event AND recorded AF diagnosis for current admission
    OR 
    (t.stay_id IS NULL AND ea.hadm_id NOT IN (SELECT hadm_id FROM current_adm_af_icd))    -- TRUE CONTROL: NO bedside event AND NO recorded AF diagnosis for current admission
  );