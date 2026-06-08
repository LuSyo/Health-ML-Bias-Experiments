
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple, cast

def class_balanced_sampling(df, class_label, n_train, n_test, seed):
  cases = df[df[class_label] == 1]
  controls = df[df[class_label] == 0]

  n_train_per_class = n_train // 2
  n_test_per_class = n_test // 2

  train_cases, test_cases = cast(
    Tuple[pd.DataFrame, pd.DataFrame],
    train_test_split(
      cases, 
      train_size=n_train_per_class, 
      test_size=n_test_per_class, 
      random_state=seed,
      shuffle=True
    ))

  train_controls, test_controls = cast(
    Tuple[pd.DataFrame, pd.DataFrame],
    train_test_split(
      controls, 
      train_size=n_train_per_class, 
      test_size=n_test_per_class, 
      random_state=seed,
      shuffle=True
    ))

  df_train = pd.concat([train_cases, train_controls]).sample(frac=1, random_state=seed)
  df_test = pd.concat([test_cases, test_controls]).sample(frac=1, random_state=seed)

  return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def simulate_selection_bias(df, n_train, n_test, seed=4):
  """
  Sub-samples the population to introduce selection bias while forcing outcome Y balance.
  Minority participants (S=0) are more likely to be selected if they are younger.
  """
  np.random.seed(seed)

  test_ratio = n_test / (n_test + n_train)
  
  # Standardize age to create stable selection weights
  age_std = (df['age'] - df['age'].mean()) / df['age'].std()
  
  # Define selection weights
  # For S=1: Selection probability increases exponentially with age
  # For S=0: Selection probability decreases slightly with age
  weights = pd.Series(np.ones(len(df)), index=df.index)
  weights[df['S'] == 1] = np.exp(0.3 * age_std[df['S'] == 1])
  weights[df['S'] == 0] = np.exp(-0.1 * age_std[df['S'] == 0])

  def get_balanced_weighted_sample(pool, count, max_ratio=1):
  # Extract weights corresponding to these specific indices
    count_per_class = count // 2
    sampled = []

    for label in [0, 1]:
      class_df = pool[pool['outcome_Y'] == label]
      n_to_sample = int(np.floor(min(len(class_df) * max_ratio, count_per_class)))
            
      if n_to_sample > 0:
        sample = class_df.sample(n=n_to_sample, 
                                weights=weights[class_df.index], 
                                random_state=seed)
        sampled.append(sample)
    
    return pd.concat(sampled)

  test_set = get_balanced_weighted_sample(df, n_test, max_ratio=test_ratio)
  
  remaining_df = df.drop(test_set.index)
  train_set = get_balanced_weighted_sample(remaining_df, n_train)

  return train_set.reset_index(drop=True), test_set.reset_index(drop=True)


def scale_dataset(df_train, df_test, minmax_features, standard_features, skewed_features):
  """
    Applies min-max normalisation and standardisation to training and test datasets
  """
  df_train_scaled = df_train.copy()
  df_test_scaled = df_test.copy()

  # Standardisation
  for var in standard_features:
    mu = df_train[var].mean()
    sigma = df_train[var].std()
    df_train_scaled[var] = (df_train[var] - mu) / sigma
    df_test_scaled[var] = (df_test[var] - mu) / sigma

  # Log Standardisation
  for var in skewed_features:
    log_mu = np.log(df_train[var]).mean()
    log_sigma = np.log(df_train[var]).std()
    df_train_scaled[var] = (np.log(df_train[var]) - log_mu) / log_sigma
    df_test_scaled[var] = (np.log(df_test[var]) - log_mu) / log_sigma

  # Normalisation
  for var in minmax_features:
    x_min = df_train[var].min()
    x_max = df_train[var].max()
    df_train_scaled[var] = (df_train[var] - x_min) / (x_max - x_min) * 3
    df_test_scaled[var] = (df_test[var] - x_min) / (x_max - x_min) * 3

  df_train_scaled.reset_index(drop=True, inplace=True)
  df_test_scaled.reset_index(drop=True, inplace=True)

  return df_train_scaled, df_test_scaled

def apply_treatment_bias(df, biomarker, s_target=0, bias_prob=0.7, b_mean_shift=-12.0, b_std_shift=3.0, seed=4):
  """
  Applies treatment bias as a treatment effect on the biomarker to everyone but individuals affected by bias

  inputs:
    - df: raw clinical dataset
    - s_target: group impacted by the bias, i.e. NOT receiving treatment
    - bias_prob: probability that the bias affects an individual in the target group
    - b_mean_shift (float): average shift for the biomarker in treated individuals
    - b_std_shift (float): shift variance for the biomarker in treated individuals

  output:
    - biased dataset
  """
  np.random.seed(seed)

  df_obs = df.copy()

  target_indices = df_obs[df_obs['S'] == s_target].index

  # Apply treatment effect to everyone but individuals affected by bias
  B_biased_indices = np.random.choice(
    target_indices, 
    size=int(len(target_indices) * bias_prob), 
    replace=False
  )
  treatment_indices = df_obs.index.difference(B_biased_indices)
  df_obs[f'{biomarker}_obs'] = df_obs[biomarker].values
  b_treatment_effect = np.random.normal(b_mean_shift, b_std_shift, size=len(treatment_indices))
  df_obs.loc[treatment_indices, f'{biomarker}_obs'] += b_treatment_effect

  return df_obs

def apply_measurement_calibration_bias(df, biomarker, s_target=0, bias_prob=1.0, added_noise_std=15.0, non_negative=False, seed=4):
  """
  Applies measurement calibration bias by injecting zero-mean Gaussian noise with 
  expanded variance strictly into individuals within the targeted demographic.
  
  inputs:
    - df: raw clinical dataset
    - biomarker: string name of the column to corrupt
    - s_target: demographic subgroup experiencing the calibration failure (e.g., S=0)
    - bias_prob: fraction of the target demographic affected by the calibration bias
    - added_noise_std: standard deviation of the injected measurement noise
    - non_negative: if True, clips values below 0 to preserve physiological realism (e.g., lab values)
    - seed: random seed for reproducibility
    
  output:
    - Copy of the dataframe with a new '{biomarker}_obs' column
  """
  np.random.seed(seed)
  df_obs = df.copy()
  
  # Initialize observed values with clean ground truth
  df_obs[f'{biomarker}_obs'] = df_obs[biomarker].values
  
  # Isolate target population indices
  target_indices = df_obs[df_obs['S'] == s_target].index
  biased_indices = np.random.choice(
      target_indices, 
      size=int(len(target_indices) * bias_prob), 
      replace=False
  )
  
  # Generate zero-mean heteroscedastic noise
  noise = np.random.normal(loc=0.0, scale=added_noise_std, size=len(biased_indices))
  df_obs.loc[biased_indices, f'{biomarker}_obs'] += noise
  
  # Enforce non-negativity constraints if mandated by the distribution family
  if non_negative:
      df_obs[f'{biomarker}_obs'] = df_obs[f'{biomarker}_obs'].clip(lower=1e-3)
      
  return df_obs

def apply_acuity_dependent_censoring(df, biomarker, s_target=0, bias_prob=0.8, threshold_quantile=0.5, attenuation_factor=0.2, seed=4):
  """
  Applies acuity-dependent censoring to simulate systemic healthcare access barriers.
  Marginalized patients (S = s_target) with low-acuity (healthy/mild) values are 
  under-sampled or have their values suppressed due to delayed clinical presentation.
  
  inputs:
    - df: raw clinical dataset
    - biomarker: string name of the continuous column to corrupt
    - s_target: demographic subgroup experiencing access barriers (S=0)
    - bias_prob: probability that a sub-acute individual is affected by the barrier
    - threshold_quantile: quantile below which an individual is considered "sub-acute"
    - attenuation_factor: multiplier applied to suppressed values (simulates downplayed severity)
    - seed: random seed for reproducibility
    
  output:
    - Copy of the dataframe with a new '{biomarker}_obs' column
  """
  np.random.seed(seed)
  df_obs = df.copy()
  
  # Initialize observed values with ground truth
  df_obs[f'{biomarker}_obs'] = df_obs[biomarker].values
  
  # Establish the empirical clinical acuity threshold from the global population
  threshold_val = df_obs[biomarker].quantile(threshold_quantile)
  
  # Identify target individuals who are medically stable/sub-acute (below threshold)
  sub_acute_target_mask = (df_obs['S'] == s_target) & (df_obs[biomarker] < threshold_val)
  subacute_indices = df_obs[sub_acute_target_mask].index
  
  # Determine which sub-acute individuals are censored/suppressed
  censored_indices = np.random.choice(
    subacute_indices, 
    size=int(len(subacute_indices) * bias_prob), 
    replace=False
  )
  
  # Suppress the values, driving them artificially lower to simulate under-coding/delayed capture
  df_obs.loc[censored_indices, f'{biomarker}_obs'] *= attenuation_factor
  
  return df_obs

def apply_constant_additive_bias(df, biomarker, s_target=0, bias_prob=1.0, shift_val=-10.0, non_negative=True, seed=4):
  """
  Applies a fixed, constant value shift to a biomarker for a specific demographic.
  Simulates a baseline calibration offset in diagnostic tools across groups.
  
  X_obs = X + shift_val  (For individuals where S == s_target)
  """
  np.random.seed(seed)
  df_obs = df.copy()
  
  # Initialize observed with clean ground truth
  df_obs[f'{biomarker}_obs'] = df_obs[biomarker].values
  
  # Isolate target indices
  target_indices = df_obs[df_obs['S'] == s_target].index
  biased_indices = np.random.choice(
    target_indices, 
    size=int(len(target_indices) * bias_prob), 
    replace=False
  )
  
  # Apply deterministic constant shift
  df_obs.loc[biased_indices, f'{biomarker}_obs'] += shift_val

  if non_negative:
    df_obs[f'{biomarker}_obs'] = df_obs[f'{biomarker}_obs'].clip(lower=1e-3)
  
  return df_obs

def apply_normal_additive_bias(df, biomarker, s_target=0, bias_prob=1.0, mean_shift=-12.0, std_shift=3.0, non_negative=True, seed=4):
  """
  Applies a random normal value shift to a biomarker for a specific demographic.
  
  X_obs = X + N(mu, std)
  """
  np.random.seed(seed)
  df_obs = df.copy()
  
  # Initialize observed with clean ground truth
  df_obs[f'{biomarker}_obs'] = df_obs[biomarker].values
  
  # Isolate target indices
  target_indices = df_obs[df_obs['S'] == s_target].index
  biased_indices = np.random.choice(
    target_indices, 
    size=int(len(target_indices) * bias_prob), 
    replace=False
  )
  
  # Apply deterministic constant shift
  add_bias = np.random.normal(loc=mean_shift, scale=std_shift, size=len(biased_indices))
  df_obs.loc[biased_indices, f'{biomarker}_obs'] += add_bias

  if non_negative:
    df_obs[f'{biomarker}_obs'] = df_obs[f'{biomarker}_obs'].clip(lower=1e-3)
  
  return df_obs


def apply_multiplicative_scaling_bias(df, biomarker, s_target=0, bias_prob=1.0, scale_factor=0.85, seed=4):
  """
  Scales the biomarker values by a fixed percentage for a specific demographic.
  Simulates a proportional under-reporting or under-measurement.
  
  X_obs = X * scale_factor  (For individuals where S == s_target)
  """
  np.random.seed(seed)
  df_obs = df.copy()
  
  # Initialize observed with clean ground truth
  df_obs[f'{biomarker}_obs'] = df_obs[biomarker].values
  
  # Isolate target indices
  target_indices = df_obs[df_obs['S'] == s_target].index
  biased_indices = np.random.choice(
      target_indices, 
      size=int(len(target_indices) * bias_prob), 
      replace=False
  )
  
  # Apply multiplicative scaling
  df_obs.loc[biased_indices, f'{biomarker}_obs'] *= scale_factor
  
  return df_obs

# ==== ARCHIVE ====

def apply_additive_unfair_bias(df, s_target=0, bias_prob=0.5, s1_max_shift=2, b1_mean_shift=12.0, b1_std_shift=3.0, proc_risk_penalty=0.15, seed=4):
  """
  Applies additive bias to clinical features

  inputs:
    - df: raw clinical dataset
    - bias_prob: probability that the bias affects an individual in the target group
    - s1_max_shift (int): maximum integer shift for the ordinal symptom_1
    - b1_mean_shift (float): average shift for biomarker 1
    - b1_std_shift (float): shift variance for biomarker 1
    - proc_risk_penalty (float): [0, 1) average reduction in risk score for referral

  output:
    - biased dataset
  """
  np.random.seed(seed)

  df_obs = df.copy()

  target_indices = df_obs[df_obs['S'] == s_target].index

  # SYMPTOM 1
  # Apply random shift between 1 and s1_max_shift to the affected individuals
  S1_biased_indices = np.random.choice(
    target_indices, 
    size=int(len(target_indices) * bias_prob), 
    replace=False
  )
  df_obs['symptom_1_obs'] = df_obs['symptom_1'].values
  s1_random_shifts = np.random.randint(1, s1_max_shift + 1, size=len(S1_biased_indices))
  df_obs.loc[S1_biased_indices, 'symptom_1_obs'] = np.clip(
    df_obs.loc[S1_biased_indices, 'symptom_1_obs'] - s1_random_shifts, 0, 4
  )

  # BIOMARKER 1
  # Apply treatment effect to everyone but individuals affected by bias
  B1_biased_indices = np.random.choice(
    target_indices, 
    size=int(len(target_indices) * bias_prob), 
    replace=False
  )
  treatment_indices = df_obs.index.difference(B1_biased_indices)
  df_obs['biomarker_1_obs'] = df_obs['biomarker_1'].values
  b1_treatment_effect = np.random.normal(b1_mean_shift, b1_std_shift, size=len(treatment_indices))
  df_obs.loc[treatment_indices, 'biomarker_1_obs'] -= b1_treatment_effect

  # PROCEDURE
  proc_biased_indices = np.random.choice(
    target_indices, 
    size=int(len(target_indices) * bias_prob), 
    replace=False
  )
  df_obs['procedure'] = (df_obs['risk_score_proc'] > 6).astype(int)
  proc_penalties = np.random.uniform(0, proc_risk_penalty * 2, size=len(proc_biased_indices))
  biased_risk = df_obs.loc[proc_biased_indices, 'risk_score_proc'] - proc_penalties
  df_obs.loc[proc_biased_indices, 'procedure'] = (biased_risk > 6).astype(int)

  return df_obs



def apply_systemic_biases(df, reporting_threshold=2, medication_effect=15.0, referral_bias=0.3, missingness_rate=0.4):
  """
  Introduces unfair pathways into the observed clinical features.
  """
  # Create observed copies to preserve 'true' columns for the audit
  S = df['S'].values

  # 1. Symptom 1 (Reporting Bias): High values in S=0 are recorded lower
  # If reported > threshold, it's 'capped' or reduced for the minority group
  df['symptom_1_obs'] = df['symptom_1']
  mask_s1 = (S == 0) & (df['symptom_1_obs'] > reporting_threshold)
  df.loc[mask_s1, 'symptom_1_obs'] = df.loc[mask_s1, 'symptom_1_obs'] - 1

  # 2. Biomarker 1 (Treatment Bias): S=1 has higher access to regulating meds
  # Medication typically 'regulates' (lowers) a high biomarker toward normal range
  # We apply this only to the Majority group to simulate differential access
  df['biomarker_1_obs'] = df['biomarker_1']
  df.loc[S == 1, 'biomarker_1_obs'] = df.loc[S == 1, 'biomarker_1_obs'] - medication_effect

  # 3. Procedure (Referral Bias): S=0 less often referred despite high risk
  # We define a 'referred' binary variable based on the risk_score_proc
  # Majority (S=1) systematically referred if score > 0.5; 
  # Minority (S=0) needs score > 0.5 but referral are systematically denied
  base_referral = df['risk_score_proc'] > 0.5
  bias_mask = (S == 0) & (np.random.random(len(df)) < referral_bias)
  df['procedure'] = base_referral.astype(int)
  df.loc[bias_mask, 'procedure'] = 0 # Systemic denial of referral

  # 4. Symptom 3 (Mixed Pathway Bias): S=0 under-reports/never reports
  # A proportion of 'True' symptom_3 cases in group 0 are zeroed out
  df['symptom_3_obs'] = df['symptom_3']
  s3_report_mask = (S == 0) & (df['symptom_3_obs'] == 1) & (np.random.random(len(df)) < missingness_rate)
  df.loc[s3_report_mask, 'symptom_3_obs'] = 0

  df_obs = df.drop(['symptom_1', 'biomarker_1', 'risk_score_proc', 'symptom_3'], axis=1)

  return df_obs