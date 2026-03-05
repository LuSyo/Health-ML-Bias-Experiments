import numpy as np
import pandas as pd
import pytest
from src.metrics import calculate_performance_metrics, stratified_perf
from src.causal_validation import calculate_te_error, latent_recon_loss

def test_calculate_performance_metrics():
    # 1. Setup mock data
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.6, 0.7, 0.3])

    # 2. Execute
    metrics = calculate_performance_metrics(y_true, y_pred, y_prob)

    # 3. Assert (based on manual calculation)
    # TP=3, FP=1, FN=1, TN=3 -> Accuracy = 6/8 = 0.75
    # FNR = FN / (FN+TP) = 1 / (1+3) = 0.25
    # FPR = FP / (FP+TN) = 1 / (1+3) = 0.25
    assert metrics['accuracy'] == 0.75
    assert metrics['fnr'] == 0.25
    assert metrics['fpr'] == 0.25
    assert 'roc_auc' in metrics
    assert 'brier_score' in metrics

def test_stratified_perf():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.1, 0.4, 0.9, 0.8, 0.2])
    sens =   np.array([0, 0, 0, 0, 1, 1, 1, 1]) # 4 in group 0, 4 in group 1

    df_strat = stratified_perf(y_true, y_pred, y_prob, sens)

    # Group 0: y_true=[1,0,1,0], y_pred=[1,0,1,0] -> Acc = 1.0, FNR = 0.0
    # Group 1: y_true=[1,0,1,0], y_pred=[0,1,1,0] -> Acc = 0.5, FNR = 0.5
    assert df_strat.loc[df_strat['group'] == 0, 'accuracy'].values[0] == 1.0
    assert df_strat.loc[df_strat['group'] == 0, 'fnr'].values[0] == 0.0
    assert df_strat.loc[df_strat['group'] == 1, 'accuracy'].values[0] == 0.5
    assert df_strat.loc[df_strat['group'] == 1, 'fnr'].values[0] == 0.5

def test_calculate_te_error():
    # 1. Setup mock data to simulate a specific disparity and counterfactual shift
    y_true = np.array(      [1,   1,   0,   0,   1,   1,   1,   1])
    sens = np.array(        [0,   0,   0,   0,   1,   1,   1,   1])
    # Group 0 mean = 0.5
    # Group 1 mean = 1.0
    # Expected Observed Disparity = 1.0 - 0.5 = 0.5

    y_pred_prob = np.array( [0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8])
    y_cf_prob = np.array(   [0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5])
    
    # y_do_1 (set S=1): for S=0 patients it's cf (0.6), for S=1 patients it's factual (0.8)
    # y_do_1 mean = (0.6*4 + 0.8*4) / 8 = 0.7
    # y_do_0 (set S=0): for S=0 patients it's factual (0.4), for S=1 patients it's cf (0.5)
    # y_do_0 mean = (0.4*4 + 0.5*4) / 8 = 0.45
    # Expected ATE = 0.7 - 0.45 = 0.25
    # Expected TE Error = abs(0.5 - 0.25) = 0.25

    te_error, obs_disp, est_ate = calculate_te_error(y_true, y_pred_prob, y_cf_prob, sens)

    assert np.isclose(obs_disp, 0.5)
    assert np.isclose(est_ate, 0.25)
    assert np.isclose(te_error, 0.25)

def test_latent_recon_loss():
    u = np.array([[1.0, 2.0], [3.0, 4.0]])
    u_cf = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Exact match should yield 0 loss
    assert latent_recon_loss(u, u_cf) == 0.0

    u_cf_diff = np.array([[2.0, 2.0], [3.0, 5.0]])
    # Dim 0: MSE of [1,3] vs [2,3] -> ((1-2)^2 + (3-3)^2)/2 = 0.5
    # Dim 1: MSE of [2,4] vs [2,5] -> ((2-2)^2 + (4-5)^2)/2 = 0.5
    # Mean across dims = 0.5
    assert latent_recon_loss(u, u_cf_diff) == 0.5