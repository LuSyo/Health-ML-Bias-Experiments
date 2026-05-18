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