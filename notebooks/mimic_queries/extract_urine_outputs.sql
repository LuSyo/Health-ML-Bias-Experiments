WITH raw_urine_flows AS (
  SELECT 
    co.stay_id,
    -- Standard Urine Output items
    SUM(CASE WHEN oe.itemid IN (226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584) 
             THEN TRY_CAST(oe.value AS NUMERIC) ELSE 0 END) AS standard_urine,
    
    -- Gross drainage out (Urine + Irrigant fluid)
    SUM(CASE WHEN oe.itemid = 227489 
             THEN TRY_CAST(oe.value AS NUMERIC) ELSE 0 END) AS gross_irrigant_out,
    
    -- Active irrigant fluid volume infused into the bladder
    SUM(CASE WHEN oe.itemid = 227488 
             THEN TRY_CAST(oe.value AS NUMERIC) ELSE 0 END) AS irrigant_in
  FROM cohort co
  INNER JOIN outputevents oe ON co.stay_id = oe.stay_id
  WHERE oe.charttime BETWEEN co.intime AND co.intime + INTERVAL 1 DAY
  GROUP BY co.stay_id
),

net_urine_calculated AS (
  SELECT 
    stay_id,
    -- Net urine calculation: Standard output + (Gross Drain Out - Infused Saline In)
    GREATEST(0, standard_urine + (gross_irrigant_out - irrigant_in)) AS net_fluid_output
  FROM raw_urine_flows
)

SELECT 
  co.stay_id,
  -- Hard physiological bounding applied directly to the final net sum
  -- If the calculation fails to bring the value under 10 Liters, suppress it to NULL
  CASE 
    WHEN n.net_fluid_output BETWEEN 0 AND 10000 THEN n.net_fluid_output
    ELSE NULL 
  END AS urine_output_total
FROM cohort co
LEFT JOIN net_urine_calculated n ON co.stay_id = n.stay_id;
