import numpy as np
import pandas as pd
from typing import cast
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture

class StrataAwareRobustScaler(BaseEstimator, TransformerMixin):
    """
    Standardizes continuous features using medians and IQRs calculated 
    independently within demographic strata to preserve baseline subgroup shifts.
    """
    def __init__(self, strata_col, continuous_cols):
      self.strata_col = strata_col
      self.continuous_cols = continuous_cols
      self.stats_ = {}
        
    def fit(self, X, y=None):
      X_df = pd.DataFrame(X).copy()
      unique_strata = X_df[self.strata_col].unique()
      
      self.stats_ = {}
      for strata in unique_strata:
        strata_mask = X_df[self.strata_col] == strata
        strata_data = cast(pd.DataFrame, X_df.loc[strata_mask, self.continuous_cols])
        
        # Compute Robust Statistics (Median and IQR) strictly within stratum
        medians = strata_data.median(axis=0)
        q25 = strata_data.quantile(0.25, axis=0)
        q75 = strata_data.quantile(0.75, axis=0)
        iqr = q75 - q25
        
        # Handle edge case: Avoid zero division if IQR is 0 for invariant features
        iqr = iqr.replace(0.0, 1.0)
        
        self.stats_[strata] = {
          'median': medians,
          'iqr': iqr
        }
      return self

    def transform(self, X):
      X_df = pd.DataFrame(X).copy()
      X_df[self.continuous_cols] = X_df[self.continuous_cols].astype(float)
      
      # Vectorized scaling across matched strata pools
      for strata, stats in self.stats_.items():
        strata_mask = X_df[self.strata_col] == strata
        if strata_mask.any():
          X_df.loc[strata_mask, self.continuous_cols] = (
              X_df.loc[strata_mask, self.continuous_cols] - stats['median']
          ) / stats['iqr']
      return X_df


class ZeroInflatedConditionalScaler(BaseEstimator, TransformerMixin):
  """
  Extracts a binary usage flag and rescales continuous residuals 
  solely on instances where an active intervention dosage exists.
  """
  def __init__(self, infusion_cols):
    self.infusion_cols = infusion_cols
    self.stats_ = {}

  def fit(self, X, y=None):
    X_df = pd.DataFrame(X).copy()
    self.stats_ = {}
    
    for col in self.infusion_cols:
      # Isolate active rows where the rate is strictly positive
      active_mask = X_df[col] > 0
      active_data = cast(pd.Series, X_df.loc[active_mask, col])
      
      if len(active_data) > 0:
        median = active_data.median()
        q25 = active_data.quantile(0.25)
        q75 = active_data.quantile(0.75)
        iqr = q75 - q25 if (q75 - q25) != 0 else 1.0
      else:
        median, iqr = 0.0, 1.0
          
      self.stats_[col] = {'median': median, 'iqr': iqr}
    return self

  def transform(self, X):
    X_df = pd.DataFrame(X).copy()
    
    for col in self.infusion_cols:
      # Generate explicit non-zero indicator mapping
      indicator_col = f"{col}_is_active"
      X_df[indicator_col] = (X_df[col] > 0).astype(float)
      
      # Apply robust transformations exclusively over positive tracks
      active_mask = X_df[col] > 0
      if active_mask.any():
        X_df.loc[active_mask, col] = (
            X_df.loc[active_mask, col] - self.stats_[col]['median']
        ) / self.stats_[col]['iqr']
          
      # Guarantee zero remains structurally absolute
      X_df.loc[~active_mask, col] = 0.0
    return X_df