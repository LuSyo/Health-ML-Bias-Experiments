
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def proportion_preserving_sampling(df, n_train, n_test, seed=42):
  """
  Sub-samples the population while strictly preserving the joint distribution of S, age_bin, and outcome_Y.
  """
  df['age_bin'] = pd.qcut(df['age'], q=10, labels=False)
  
  strat_cols = ['S', 'age_bin', 'outcome_Y']
  
  proportions = df.groupby(strat_cols).size() / len(df)
  
  test_subsets = []
  train_subsets = []
  test_ratio = n_test / (n_test + n_train)

  grouped_df = df.groupby(strat_cols)
  
  # Sample from each group according to its population weight
  for strata, prop in proportions.items():
    count_test = int(np.ceil(prop * n_test))
    count_train = int(np.ceil(prop * n_train))

    try:
      strata_df = grouped_df.get_group(strata)
    except KeyError:
      continue
    
    max_test = int(np.floor(len(strata_df)*test_ratio))
    
    test_subset = strata_df.sample(n=min(count_test, max_test), replace=False, random_state=seed)
    test_subsets.append(test_subset)

    remaining_strata = strata_df.drop(test_subset.index)
    train_subset = strata_df.sample(n=min(count_train, len(remaining_strata)), replace=False, random_state=seed)
    train_subsets.append(train_subset)

  train_out = pd.concat(train_subsets) if train_subsets else pd.DataFrame(columns=grouped_df.columns)
  test_out = pd.concat(test_subsets) if test_subsets else pd.DataFrame(columns=grouped_df.columns)
  
  return train_out.reset_index(drop=True), test_out.reset_index(drop=True)

def sample_stratified_class(population, n_train, n_test, strat_cols, proportions, class_val, seed):
  """
  Samples a class by matching given strata proportions and target class sampling size
  """
  class_df = population[population['outcome_Y'] == class_val]
  test_subsets = []
  train_subsets = []
  test_ratio = n_test / (n_test + n_train)

  grouped_class = class_df.groupby(strat_cols)

  for strata, prop in proportions.items():
    count_test = int(np.round(prop * n_test))
    count_train = int(np.round(prop * n_train))

    try:
      strata_df = grouped_class.get_group(strata)
    except KeyError:
      continue

    max_test = int(np.floor(len(strata_df)*test_ratio))
    
    test_subset = strata_df.sample(n=min(count_test, max_test), replace=False, random_state=seed)
    test_subsets.append(test_subset)

    remaining_strata = strata_df.drop(test_subset.index)
    train_subset = strata_df.sample(n=min(count_train, len(remaining_strata)), replace=False, random_state=seed)
    train_subsets.append(train_subset)

  train_out = pd.concat(train_subsets) if train_subsets else pd.DataFrame(columns=class_df.columns)
  test_out = pd.concat(test_subsets) if test_subsets else pd.DataFrame(columns=class_df.columns)
  
  return train_out.reset_index(drop=True), test_out.reset_index(drop=True)

def class_balanced_sampling(df, n_train, n_test, seed):
  df['age_bin'] = pd.qcut(df['age'], q=10, labels=False)

  strat_cols = ['S', 'age_bin']

  n_train_per_class = n_train // 2
  n_test_per_class = n_test // 2

  strat_proportions = df.groupby(strat_cols).size() / len(df)

  train_cases_sampled, test_cases_sampled = sample_stratified_class(df, n_train_per_class, n_test_per_class, strat_cols, strat_proportions, class_val=1, seed=seed)
  train_controls_sampled, test_controls_sampled, = sample_stratified_class(df, n_train_per_class, n_test_per_class, strat_cols, strat_proportions, class_val=0, seed=seed)

  train_cases_sampled.drop(columns=['age_bin'], inplace=True)
  test_cases_sampled.drop(columns=['age_bin'], inplace=True)
  train_controls_sampled.drop(columns=['age_bin'], inplace=True)
  test_controls_sampled.drop(columns=['age_bin'], inplace=True)
  df.drop(columns=['age_bin'], inplace=True)

  return pd.concat([train_cases_sampled, train_controls_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True), pd.concat([test_cases_sampled, test_controls_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)

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


def scale_dataset(df, norm_features, skewed_features):
  df_scaled = df.copy()
  # Z-score for normal features
  for var in norm_features:
    df_scaled[var] = (df_scaled[var] - df_scaled[var].mean()) / df_scaled[var].std()

  # Log and Z-score for skewed features
  for var in skewed_features:
    log_var = np.log(df[var])
    df_scaled[var] = (log_var - log_var.mean()) / log_var.std()

  df_scaled.reset_index(drop=True, inplace=True)

  return df_scaled

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

def plot_cont_feature(df, feature, label):
  fig, axes = plt.subplots(1, 2, figsize=(16, 4))
  sns.histplot(df, x=feature, hue="S", bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes[0])
  sns.histplot(df, x=feature, hue="outcome_Y", bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes[1])
  fig.suptitle(f'Probability distribution of {label}', fontsize=16)
  plt.show()

  return fig

def plot_cat_feature(df, feature, label):
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  sns.histplot(df, x=df[feature].astype(int), hue="S", common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes[0])
  sns.histplot(df, x=feature, hue="outcome_Y", common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes[1])
  fig.suptitle(f'Probability distribution of {label}', fontsize=16)
  plt.show()

  return fig


# ==== ARCHIVE ====

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