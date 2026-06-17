import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils import resample
from scipy.stats import entropy
from scipy.spatial.distance import mahalanobis, jensenshannon
from tableone import TableOne

def print_table_1(dataset, continuous_cols, categorical_cols, groupby='sex'):
  table1 = TableOne(dataset,
                    groupby=groupby,
                    continuous=continuous_cols,
                    categorical=categorical_cols,
                    missing=False
                    )

  print(table1.tabulate())

def format_probe_results(results, groups):
  columns = [c.removeprefix("u_") for c in results.filter(regex="u_.*").columns]
  formatted_results = pd.DataFrame(columns=["probe"], data=["x", "u"])
  for col in columns:
    melted_subresults = results.melt(value_vars=[f"x_{col}", f"u_{col}"], value_name=col, var_name="probe")
    melted_subresults['probe'] = melted_subresults['probe'].str[0]
    formatted_results = formatted_results.merge(melted_subresults, on="probe", how="outer")
  formatted_results.set_index("probe", inplace=True)
  formatted_results.sort_index(ascending=False, inplace=True)

  def format_score(score):
    return round(score*100, 2)

  global_results = formatted_results.filter(regex="global_.*").apply(format_score)
  group_results = []
  for g in groups:
    group_results.append(formatted_results.filter(regex=f"{g}_.*").apply(format_score))

  return global_results, group_results

def mutual_info_with_sens(X, Y, target_label, iterations=100, n_samples=None, seed=4):
  mi_results = []

  for i in range(iterations):
    X_resampled, Y_resampled = resample(X, Y, replace=False, n_samples=n_samples, random_state=i) # type: ignore
    mi_scores = mutual_info_regression(X_resampled, Y_resampled, n_neighbors=5, random_state=seed)
    mi_results.append(mi_scores)

  mi_df = pd.DataFrame(mi_results, columns=X.columns)
  mi_median = mi_df.median().sort_values(ascending=False)
  mi_df = mi_df[mi_median.index]

  print("\n--- Mutual Information with S ---\n")
  print(mi_df.describe().T.to_markdown())

  plt.figure(figsize=(6, 3))
  sns.barplot(data=mi_df, orient='h', estimator='median', palette='plasma')
  plt.title(f'Mutual Information Scores with {target_label} ({iterations} bootstrap samples)')
  plt.xlim(0, 0.5)
  plt.show()

def mutual_info_grouped(df, feature_cols, target_col, target_label, iterations=100, n_samples=None, seed=4):
  mi_results = []

  unique_patients = df['patient_index'].unique()

  for i in range(iterations):
    current_seed = seed + i
    rng = np.random.default_rng(seed=current_seed)

    # Patient sampling
    sampled_patients = rng.choice(unique_patients, size=n_samples, replace=False)
    subset_df = df[df['patient_index'].isin(sampled_patients)]

    # Keep only one random sample per patient
    shuffled_df = subset_df.sample(frac=1, random_state=current_seed)
    final_sample = shuffled_df.drop_duplicates(subset=['patient_index'], keep='first')
    
    X_resampled = final_sample[feature_cols]
    Y_resampled = final_sample[target_col]

    # Compile MI scores
    mi_scores = mutual_info_regression(X_resampled, Y_resampled, n_neighbors=5, random_state=seed)
    mi_results.append(mi_scores)

  mi_df = pd.DataFrame(mi_results, columns=feature_cols)
  mi_median = mi_df.median().sort_values(ascending=False)
  mi_df = mi_df[mi_median.index]

  print("\n--- Mutual Information with S ---\n")
  print(mi_df.describe().T.to_markdown())

  plt.figure(figsize=(6, 3))
  sns.barplot(data=mi_df, orient='h', estimator='median', palette='plasma')
  plt.title(f'Mutual Information Scores with {target_label} ({iterations} bootstrap samples)')
  plt.xlim(0, 0.5)
  plt.show()


def plot_cont_feature(df, feature, label, class_label, hue_label):
  layout = """
    AB
    CC
    """
  fig, axes = plt.subplot_mosaic(layout, figsize=(16, 9))
  sns.histplot(df, x=feature, hue=hue_label, bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes['A'])
  sns.histplot(df, x=feature, hue=class_label, bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes['B'])
  sns.histplot(df, x=feature, bins=50, kde=True, stat='probability', ax=axes['C'])
  fig.suptitle(f'Probability distribution of {label}', fontsize=16)
  plt.show()

  return fig

def plot_cat_feature(df, feature, label, class_label, hue_label):
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  # sns.histplot(df, x=df[feature].astype(int), hue=hue_label, common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes[0])
  # sns.histplot(df, x=feature, hue=class_label, common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes[1])
  sns.countplot(df, x=feature, stat="probability", hue=hue_label, ax=axes[0])
  sns.countplot(df, x=feature, stat="probability", hue=class_label, ax=axes[1])

  fig.suptitle(f'Probability distribution of {label}', fontsize=16)
  plt.show()

  return fig

def plot_cf_cont_feature_comparison(df, feature_col, cf_feature_col, sens_col, target_col):
  long_df = df.melt(id_vars=[sens_col, target_col], value_vars=[feature_col, cf_feature_col], var_name="feature")
  
  # compare CF feature of patients originally from group 0 with factual feature of patients originally from group 1, and vice versa
  # => flip sens for the CF feature values
  long_df['new_sens'] = np.where(
    long_df['feature'] == feature_col, 
    long_df[sens_col], 
    1-long_df[sens_col])
  
  group_0_mask = long_df['new_sens'] == 0
  group_1_mask = long_df['new_sens'] == 1
  target_neg = long_df[target_col] == 0
  target_pos = long_df[target_col] == 1

  fig, axes = plt.subplots(2, 2, figsize=(18, 10))
  sns.histplot(long_df[group_0_mask & target_neg], x="value", hue="feature", bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes[0, 0])
  sns.histplot(long_df[group_0_mask & target_pos], x="value", hue="feature", bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes[0, 1])
  sns.histplot(long_df[group_1_mask & target_neg], x="value", hue="feature", bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes[1, 0])
  sns.histplot(long_df[group_1_mask & target_pos], x="value", hue="feature", bins=50, common_norm=False, multiple='dodge', kde=True, stat='probability', ax=axes[1, 1])

  axes[0, 0].set_xlabel(f"{feature_col} for S group 0, negative target")
  axes[1, 0].set_xlabel(f"{feature_col} for S group 1, negative target")
  axes[0, 1].set_xlabel(f"{feature_col} for S group 0, positive target")
  axes[1, 1].set_xlabel(f"{feature_col} for S group 1, positive target")
  plt.show()

def plot_cf_cat_feature_comparison(df, feature_col, cf_feature_col, sens_col, target_col):
  long_df = df.melt(id_vars=[sens_col, target_col], value_vars=[feature_col, cf_feature_col], var_name="feature")
  
  # compare CF feature of patients originally from group 0 with factual feature of patients originally from group 1, and vice versa
  # => flip sens for the CF feature values
  long_df['new_sens'] = np.where(
    long_df['feature'] == feature_col, 
    long_df[sens_col], 
    1-long_df[sens_col])
  
  group_0_mask = long_df['new_sens'] == 0
  group_1_mask = long_df['new_sens'] == 1
  target_neg = long_df[target_col] == 0
  target_pos = long_df[target_col] == 1

  cat_n = len(long_df["value"].unique())

  fig, axes = plt.subplots(2, 2, figsize=(cat_n*1.5 + 4, 10))
  sns.histplot(long_df[group_0_mask & target_neg], x="value", hue="feature", discrete=True, common_norm=False, multiple='dodge', stat='probability', ax=axes[0, 0])
  sns.histplot(long_df[group_0_mask & target_pos], x="value", hue="feature", discrete=True, common_norm=False, multiple='dodge', stat='probability', ax=axes[0, 1])
  sns.histplot(long_df[group_1_mask & target_neg], x="value", hue="feature", discrete=True,common_norm=False, multiple='dodge', stat='probability', ax=axes[1, 0])
  sns.histplot(long_df[group_1_mask & target_pos], x="value", hue="feature", discrete=True,common_norm=False, multiple='dodge', stat='probability', ax=axes[1, 1])
  axes[0, 0].set_xlabel(f"{feature_col} for S group 0, negative target")
  axes[1, 0].set_xlabel(f"{feature_col} for S group 1, negative target")
  axes[0, 1].set_xlabel(f"{feature_col} for S group 0, positive target")
  axes[1, 1].set_xlabel(f"{feature_col} for S group 1, positive target")
  plt.show()


def analyse_cont_features_disparity(df, continuous_cols, sens_col='sex', label_col='cvd'):
  """
  Computes stratified continuous feature variances, conditional variances,
  and the Mahalanobis Distance between demographic subgroups.
  """
  
  print("="*60)
  print(" 1. RAW FEATURE VARIANCE BY GROUP")
  print("="*60)
  global_var = df.groupby(sens_col)[continuous_cols].var()
  print(global_var)
  
  print("\n" + "="*60)
  print(" 2. CONDITIONAL VARIANCE DISPARITY (stratified by target)")
  print("="*60)
  conditional_var = df.groupby([sens_col, label_col])[continuous_cols].var()
  print(conditional_var)
  
  # 3. Compute Multivariate Disparity via Mahalanobis Distance
  print("\n" + "="*60)
  print(" 3. MULTIVARIATE DISTANCE (Mahalanobis Distance between Groups)")
  print("="*60)
  
  # Split datasets
  group_0 = df[df[sens_col] == 0][continuous_cols]
  group_1 = df[df[sens_col] == 1][continuous_cols]
  
  # Calculate pooled inverse covariance matrix
  cov_0 = np.cov(group_0.values.T)
  cov_1 = np.cov(group_1.values.T)
  # weighted average of the two covariance matrices based on the sample size of each subgroup
  pooled_cov = (cov_0 * len(group_0) + cov_1 * len(group_1)) / (len(group_0) + len(group_1))
  inv_pooled_cov = np.linalg.inv(pooled_cov)
  
  mean_0 = group_0.mean().values
  mean_1 = group_1.values.mean(axis=0)
  
  m_dist = mahalanobis(mean_0, mean_1, inv_pooled_cov)
  print(f"Overall Mahalanobis Distance between Group 0 and Group 1: {m_dist:.4f}")
  
  # Stratified Mahalanobis for conditional confounding analysis
  for target in df[label_col].unique():
    sub_g0 = df[(df[sens_col] == 0) & (df[label_col] == target)][continuous_cols]
    sub_g1 = df[(df[sens_col] == 1) & (df[label_col] == target)][continuous_cols]
    
    p_cov = (np.cov(sub_g0.values.T) * len(sub_g0) + np.cov(sub_g1.values.T) * len(sub_g1)) / (len(sub_g0) + len(sub_g1))
    inv_p_cov = np.linalg.inv(p_cov)
    
    m_dist_strat = mahalanobis(sub_g0.mean().values, sub_g1.mean().values, inv_p_cov)
    print(f" -> Mahalanobis Distance given CVD={target}: {m_dist_strat:.4f}")

def compute_multivariate_categorical_jsd(df, categorical_cols, protected_col='sex', label_col='cvd'):
  """
  Computes a single, global Multivariate Jensen-Shannon Divergence 
  across the joint distribution of all categorical features.
  """
  
  print("="*70)
  print(" MULTIVARIATE JOINT JENSEN-SHANNON DIVERGENCE")
  print(" (Evaluates interactions and combinations across all categorical features)")
  print("="*70)
  
  # Step 1: Serialize the combinations into a single string/tuple representation
  # This creates a composite 'joint feature' representing the multivariate state
  df_joint = df.copy()
  df_joint['joint_state'] = df_joint[categorical_cols].astype(str).agg('-'.join, axis=1)
  all_possible_states = sorted(df_joint['joint_state'].unique())
  
  # Helper to calculate normalized probability vector for a given dataframe subset
  def get_joint_probability_vector(sub_df):
    counts = sub_df['joint_state'].value_counts(normalize=True)
    # Map to a static vector aligned across all possible states
    return np.array([counts.get(state, 0.0) for state in all_possible_states])
  
  # Compute Global Multivariate Disparity
  p_global = get_joint_probability_vector(df_joint[df_joint[protected_col] == 0])
  q_global = get_joint_probability_vector(df_joint[df_joint[protected_col] == 1])
  
  jsd_global = jensenshannon(p_global, q_global)
  print(f"Global Multivariate JSD (Female vs Male): {jsd_global:.4f}")
  
  # Compute Stratified Multivariate Disparity (Conditional Causal Check)
  print("\nStratified by Clinical Outcome (CVD):")
  for target in sorted(df_joint[label_col].unique()):
    g0_strat = df_joint[(df_joint[protected_col] == 0) & (df_joint[label_col] == target)]
    g1_strat = df_joint[(df_joint[protected_col] == 1) & (df_joint[label_col] == target)]
    
    p_strat = get_joint_probability_vector(g0_strat)
    q_strat = get_joint_probability_vector(g1_strat)
    
    jsd_strat = jensenshannon(p_strat, q_strat)
    print(f" -> Given CVD={target}: Multivariate JSD = {jsd_strat:.4f}")

def compute_mixed_interaction_mahalanobis(df, continuous_subs, categorical_subs, sens_col='sex'):
  """
  Computes Mahalanobis Distance between protected groups, stratified by 
  the joint-states of a targeted subselection of categorical variables.
  """
  print("="*80)
  print(" STRATIFIED MULTIVARIATE MAHALANOBIS VIA CATEGORICAL JOINT-STATES")
  print("="*80)
  
  df_mixed = df.copy()

  # Create the joint categorical state string
  df_mixed['cat_state'] = df_mixed[categorical_subs].astype(str).agg('-'.join, axis=1)
  
  unique_states = df_mixed['cat_state'].unique()
  
  for state in unique_states:
    # Isolate the specific categorical subgroup
    sub_df = df_mixed[df_mixed['cat_state'] == state]
    
    g0 = sub_df[sub_df[sens_col] == 0][continuous_subs]
    g1 = sub_df[sub_df[sens_col] == 1][continuous_subs]
    
    # Enforce a minimum sample size check to prevent singular covariance matrices
    if len(g0) < len(continuous_subs) + 1 or len(g1) < len(continuous_subs) + 1:
        print(f"Category State [{state}]: Skipped due to insufficient sample size (Females: {len(g0)}, Males: {len(g1)})")
        continue
        
    # Compute pooled covariance and regularized pseudo-inverse
    cov_0 = np.cov(g0.values.T)
    cov_1 = np.cov(g1.values.T)
    pooled_cov = (cov_0 * len(g0) + cov_1 * len(g1)) / (len(g0) + len(g1))
    
    # Handle 1D array cases if only one continuous variable is passed
    if len(continuous_subs) == 1:
        mean_diff = abs(g0.mean().iloc[0] - g1.mean().iloc[0])
        pooled_variance = pooled_cov if isinstance(pooled_cov, (int, float)) else pooled_cov.item()
        
        if pooled_variance > 0:
            m_dist = mean_diff / np.sqrt(pooled_variance)
        else:
            m_dist = 0.0
    else:
        inv_pooled_cov = np.linalg.pinv(pooled_cov)
        m_dist = mahalanobis(g0.mean().values, g1.mean().values, inv_pooled_cov)
        
    print(f"Category State [{state}] -> Continuous Mahalanobis Distance: {m_dist:.4f} (n_female={len(g0)}, n_male={len(g1)})")

def compute_subgroup_entropy(df, continuous_subs, categorical_subs, sens_col="sex", n_bootstrap=200, seed=4):
  """
  COmputes internal consistency using normalised joint Shannon entropy after continuous feature binning
  """
  np.random.seed(seed)
  
  print("="*75)
  print(" MULTIVARIATE ENTROPY-BASED CONSISTENCY (1 = Purely Predictable/Consistent)")
  print("="*75)

  df_entropy = df.copy()

  # Bin the continuous features
  binned_cols = []
  for col in continuous_subs:
    bin_name = f"{col}_binned"
    df_entropy[bin_name] = pd.qcut(df_entropy[col], q=5, labels=False)
    binned_cols.append(bin_name)

  all_target_cols = categorical_subs + binned_cols
  df_entropy['joint_state'] = df_entropy[all_target_cols].astype(str).agg('-'.join, axis=1)

  # Identify the minority
  counts = df_entropy[sens_col].value_counts()
  minority_val = counts.idxmin()
  majority_val = counts.idxmax()
  
  n_minority = counts.min()

  def get_raw_entropy(series):
    probs = series.value_counts(normalize=True)
    return entropy(probs, base=2)

  # Minority entropy
  minority_series = df_entropy[df_entropy[sens_col] == minority_val]['joint_state']
  entropy_minority = get_raw_entropy(minority_series)

  # Majority bootstrapped entropy, drawing minority-sized subsamples of the majority 
  majority_pool = df_entropy[df_entropy[sens_col] == majority_val]['joint_state'].values
  bootstrapped_entropies = []
  
  for _ in range(n_bootstrap):
    sample = np.random.choice(majority_pool, size=n_minority, replace=False)
    bootstrapped_entropies.append(get_raw_entropy(pd.Series(sample)))

  entropy_majority_corrected = np.mean(bootstrapped_entropies)
  entropy_majority_std = np.std(bootstrapped_entropies)

  # Final summary
  unique_global_states = df_entropy['joint_state'].nunique()
  max_entropy = np.log2(unique_global_states) if unique_global_states > 1 else 1.0

  norm_entropy_minority = entropy_minority / max_entropy
  norm_entropy_majority = entropy_majority_corrected / max_entropy

  print("="*70)
  print(" BIAS-CORRECTED MULTIVARIATE ENTROPY AUDIT")
  print("="*70)
  print(f"Minority Subgroup [{sens_col}={minority_val}] (True n={n_minority}):")
  print(f" -> True Joint Shannon Entropy : {entropy_minority:.4f} bits")
  print(f" -> Empirical Consistency Score : {1.0 - norm_entropy_minority:.4f}")
  
  print(f"\nMajority Subgroup [{sens_col}={majority_val}] (Sample-Size Matched to n={n_minority}):")
  print(f" -> Mean Bootstrapped Entropy  : {entropy_majority_corrected:.4f} bits (±{entropy_majority_std:.4f})")
  print(f" -> Empirical Consistency Score : {1.0 - norm_entropy_majority:.4f}")
  print("="*70)

def cat_features_jsd(df, categorical_cols, sens_col='sex', label_col='cvd'):
  """
  Computes Jensen-Shannon Divergence for categorical features 
  between demographic groups, stratified by the disease target.
  """
  
  print("="*60)
  print(" JENSEN-SHANNON DIVERGENCE FOR CATEGORICAL DISPARITY")
  print(" (0 = Identical Distributions, 1 = Complete Disparity)")
  print("="*60)
  
  for col in categorical_cols:
    print(f"\nFeature: [{col}]")
    
    # Global Disparity
    p_ch = df[df[sens_col] == 0][col].value_counts(normalize=True)
    q_ch = df[df[sens_col] == 1][col].value_counts(normalize=True)
    
    # Align indexes to ensure missing categories match zeros
    all_cats = sorted(list(set(df[col].dropna().unique())))
    p = np.array([p_ch.get(cat, 0.0) for cat in all_cats])
    q = np.array([q_ch.get(cat, 0.0) for cat in all_cats])
    
    js_global = jensenshannon(p, q)
    print(f" -> Global JS Divergence (Female vs Male): {js_global:.4f}")
    
    # Stratified Disparity (Causal Confounding Check)
    for target in sorted(df[label_col].unique()):
      sub_0 = df[(df[sens_col] == 0) & (df[label_col] == target)][col].value_counts(normalize=True)
      sub_1 = df[(df[sens_col] == 1) & (df[label_col] == target)][col].value_counts(normalize=True)
      
      p_strat = np.array([sub_0.get(cat, 0.0) for cat in all_cats])
      q_strat = np.array([sub_1.get(cat, 0.0) for cat in all_cats])
      
      js_strat = jensenshannon(p_strat, q_strat)
      print(f"    -> Given CVD={target}: JS Divergence = {js_strat:.4f}")