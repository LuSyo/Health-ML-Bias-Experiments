import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils import resample

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
  sns.histplot(df, x=df[feature].astype(int), hue=hue_label, common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes[0])
  sns.histplot(df, x=feature, hue=class_label, common_norm=False, multiple='dodge', discrete=True, stat='probability', ax=axes[1])
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