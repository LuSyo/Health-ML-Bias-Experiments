import duckdb
import os
from config import Config

db_path = os.path.join(Config.DATA_DIR, 'mimic_local.db')
conn = duckdb.connect(db_path)

MIMIC_PATH = os.path.join(Config.DATA_DIR, "mimic_iv/physionet.org/files/mimiciv/3.1")
MIMIC_ICU_PATH = f"{MIMIC_PATH}/icu"
MIMIC_HOSP_PATH = f"{MIMIC_PATH}/hosp"

print("Materializing baseline demographics and administrative tables...")
conn.execute(f"CREATE OR REPLACE TABLE icustays AS SELECT * FROM read_csv_auto('{MIMIC_ICU_PATH}/icustays.csv.gz')")
conn.execute(f"CREATE OR REPLACE TABLE admissions AS SELECT * FROM read_csv_auto('{MIMIC_HOSP_PATH}/admissions.csv.gz')")
conn.execute(f"CREATE OR REPLACE TABLE patients AS SELECT * FROM read_csv_auto('{MIMIC_HOSP_PATH}/patients.csv.gz')")

print("Extracting target-specific ICD Diagnoses (Pre-existing history check)...")
conn.execute(f"""
    CREATE OR REPLACE TABLE diagnoses_icd AS 
    SELECT subject_id, hadm_id, icd_code, icd_version, 
        REPLACE(icd_code, '.', '') AS clean_code 
    FROM read_csv_auto('{MIMIC_HOSP_PATH}/diagnoses_icd.csv.gz')
    WHERE clean_code = '42731' OR clean_code LIKE 'I48%' -- AF
        OR clean_code LIKE '9959%' OR clean_code LIKE 'A41%' -- Sepsis
        -- hypertension
        OR clean_code IN ('I10', 'I11', 'I12', 'I13', 'I15', 'N262') 
        OR clean_code IN ('4011', '4019', '40210', '40290', '40410', '40490', '40511', '40519', '40591', '40599')
        -- diabetes mellitus
        OR LEFT(clean_code, 3) IN ('E10', 'E11', 'E12', 'E13', 'E14')
        OR clean_code LIKE '250%'
        -- CKD
        OR clean_code LIKE 'N18%' OR clean_code LIKE '585%'
        -- Old infarction
        OR clean_code = 'I252' OR clean_code = '412'
        -- Chronic Lung Disease
        OR LEFT(clean_code, 3) IN ('J41', 'J42', 'J43', 'J44')
        OR LEFT(clean_code, 3) IN ('491', '492', '496')
        -- Chronic liver disease
        OR clean_code LIKE 'K70%' OR clean_code LIKE 'K74%'
        OR clean_code LIKE '571%' OR clean_code LIKE '572%'
        -- Alcohol use
        OR clean_code IN ('V113', 'E52', 'G621', 'I426', 'K292', 'T51', 'Z714', 'Z658') 
        OR LEFT(clean_code, 3) IN ('F10', 'K70')
        OR clean_code IN ('30393', '30503') OR LEFT(clean_code, 3) = '291' 
""")

print("Extracting target-specific ICD Procedures (Cardiac surgery exclusions)...")
# Keep only rows that could match CABG or Valve procedures to save space
conn.execute(f"""
    CREATE OR REPLACE TABLE procedures_icd AS 
    SELECT subject_id, hadm_id, icd_code 
    FROM read_csv_auto('{MIMIC_HOSP_PATH}/procedures_icd.csv.gz')
    WHERE icd_code LIKE '361%' OR icd_code LIKE '351%' OR icd_code LIKE '352%'
       OR icd_code LIKE '021%' OR icd_code LIKE '02R%'
""")

print("Filtering and materializing specialized chart events (This will take a few minutes)...")
conn.execute(f"""
    CREATE OR REPLACE TABLE chartevents AS 
    SELECT stay_id, charttime, itemid, valuenum, value
    FROM read_csv_auto('{MIMIC_ICU_PATH}/chartevents.csv.gz')
    WHERE itemid IN (
        220048, -- Telemetry (AF checks)
        220045, -- Heart Rate
        220179, 220059, -- SBP
        220180, 220060, -- DBP
        220052, 220181, -- Blended Mean Blood Pressure
        223761, 223762, -- Temperature
        220277, -- SpO2
        226512, 224639, 226846, -- Weight items
        223835, -- FiO2 (Bedside Respiratory Support Component)
        220210, -- Respiratory Rate
        220739, -- GCS Eye Opening
        223900, -- GCS Verbal Response
        223901  -- GCS Motor Response
    )
""")

print("Filtering and materialising specialised lab events...")
conn.execute(f"""
    CREATE OR REPLACE TABLE labevents AS
    SELECT hadm_id, itemid, valuenum, charttime
    FROM read_csv_auto('{MIMIC_HOSP_PATH}/labevents.csv.gz')
    WHERE itemid IN (
        51222, --hemoglobin
        51300, 51301, --white blood cells
        51265, --platelets
        51006, --BUN
        50912, --creatinine
        50809, 50931, --glucose
        50868, --anion gap
        50971, 50822, --potassium
        50983, 50824, --sodium
        50893, --calcium
        50910, --CK CPK
        50911, -- CK MB
        50963, --NT proBNP
        50821, -- PaO2 (SOFA Respiratory raw metric)
        50885  -- Total Bilirubin (SOFA Liver metric)
    )
""")

print("Filtering and materialising specialised output events...")
conn.execute(f"""
    CREATE OR REPLACE TABLE outputevents AS
    SELECT stay_id, charttime, itemid, value
    FROM read_csv_auto('{MIMIC_ICU_PATH}/outputevents.csv.gz')
    -- Urine outputs
    WHERE itemid IN (226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227489, 227488)
""")

print("Filtering and materialising specialised procedure events...")
conn.execute(f"""
    CREATE OR REPLACE TABLE procedureevents AS
    SELECT stay_id, itemid, starttime
    FROM read_csv_auto('{MIMIC_ICU_PATH}/procedureevents.csv.gz')
    WHERE itemid IN (
        225792, 225794, 224385, --Mechanical Ventilation
        225441, 225802, 225803, 225805, 224149, 225809 --CRRT
    )
""")

print("Filtering and materialising specialised input events...")
conn.execute(f"""
    CREATE OR REPLACE TABLE inputevents AS
    SELECT stay_id, itemid, starttime, endtime, rate, rateuom
    FROM read_csv_auto('{MIMIC_ICU_PATH}/inputevents.csv.gz')
    WHERE itemid IN (
        221906, 229617, 221289, 221749, 222315, 221662 --Vasopressors
    )
""")

print("Filtering and materialising specialized prescription events (Antibiotics)...")
conn.execute(f"""
    CREATE OR REPLACE TABLE prescriptions AS
    SELECT hadm_id, starttime, stoptime, drug
    FROM read_csv_auto('{MIMIC_HOSP_PATH}/prescriptions.csv.gz')
    WHERE route IN ('IV', 'IM', 'INJ') -- Target systemic administrations
      AND (
        LOWER(drug) LIKE '%cef%' OR LOWER(drug) LIKE '%cillin%' OR 
        LOWER(drug) LIKE '%penem%' OR LOWER(drug) LIKE '%floxacin%' OR
        LOWER(drug) LIKE '%mycin%' OR LOWER(drug) LIKE '%bactam%' OR
        LOWER(drug) LIKE '%metronidazole%' OR LOWER(drug) LIKE '%linezolid%'
      )
""")

print("Database initialization complete! Local database saved as 'mimic_local.db'")
conn.close()