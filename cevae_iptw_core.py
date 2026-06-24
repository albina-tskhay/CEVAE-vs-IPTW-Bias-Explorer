"""
cevae_iptw_core.py
Pure computation layer for the CEVAE vs IPTW sensitivity check.
No Streamlit dependency, so this can be imported and tested independently
of the app UI, and reused if the app is later rehosted on a different
framework (e.g. Gradio).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import torch
import pyro
from joblib import Parallel, delayed
import os

from cevae_test_mod import CEVAE

# Fast defaults. The original app used hidden_dim=140, num_layers=3 (matching
# a research benchmark architecture) with num_epochs=100 and no early
# stopping; a single fit at that size took ~55s on a 4000-row, 4-proxy
# dataset. These defaults trade some representational capacity for speed,
# appropriate for an interactive sensitivity-check tool rather than a
# benchmark run.
DEFAULT_HIDDEN_DIM = 20
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_EPOCHS = 80
DEFAULT_LEARNING_RATE = 1e-3


def fit_cevae_once(x, t, y, feature_dim, outcome_dist,
                    hidden_dim=DEFAULT_HIDDEN_DIM, num_layers=DEFAULT_NUM_LAYERS,
                    num_epochs=DEFAULT_NUM_EPOCHS, batch_size=200,
                    learning_rate=DEFAULT_LEARNING_RATE, seed=0):
    """Fit CEVAE once on tensors x, t, y and return the estimated ATE."""
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    cevae = CEVAE(
        feature_dim=feature_dim,
        outcome_dist=outcome_dist,
        latent_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_samples=100,
    )
    cevae.fit(
        x, t, y,
        num_epochs=num_epochs,
        batch_size=min(batch_size, len(x)),
        learning_rate=learning_rate,
        learning_rate_decay=0.5,
        weight_decay=1e-4,
        log_every=0,
    )
    with torch.no_grad():
        ite = cevae.ite(x).cpu().numpy()
    return float(ite.mean())


def fit_iptw(proxies_df, t, y):
    """Fit IPTW via logistic-regression propensity scores. Returns (ate, (ci_low, ci_high))."""
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(proxies_df, t)
    ps = np.clip(ps_model.predict_proba(proxies_df)[:, 1], 0.02, 0.98)
    weights = t / ps + (1 - t) / (1 - ps)
    model = sm.WLS(y, sm.add_constant(t), weights=weights).fit()
    ate = float(model.params.iloc[1])
    se = float(model.bse.iloc[1])
    return ate, (ate - 1.96 * se, ate + 1.96 * se)


def bootstrap_iptw(x_np, t_np, y_np, n_boot, n_jobs, seed=0):
    n = len(x_np)
    rng = np.random.default_rng(seed)

    def one_run(i):
        idx = rng.integers(0, n, n)
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(x_np[idx], t_np[idx])
        ps = np.clip(ps_model.predict_proba(x_np[idx])[:, 1], 0.02, 0.98)
        w = t_np[idx] / ps + (1 - t_np[idx]) / (1 - ps)
        m = sm.WLS(y_np[idx], sm.add_constant(t_np[idx]), weights=w).fit()
        return float(m.params[1])

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one_run)(i) for i in range(n_boot)
    )
    return np.array(results)


def bootstrap_cevae(x_np, t_np, y_np, outcome_dist, n_boot, n_jobs, seed=0,
                     hidden_dim=DEFAULT_HIDDEN_DIM, num_layers=DEFAULT_NUM_LAYERS,
                     num_epochs=DEFAULT_NUM_EPOCHS):
    """Parallel bootstrap for CEVAE: refits the full model on each resample.
    This is the step that was serial and unparallelized in the original app
    despite importing joblib.Parallel; here it actually uses it, via real
    processes rather than threads. CEVAE fitting is CPU-bound (PyTorch
    holds the GIL for most of its work), so thread-based "parallelism"
    would not give a real speedup; only process-based parallelism does."""
    n = len(x_np)
    rng = np.random.default_rng(seed)
    feature_dim = x_np.shape[1]

    def one_run(i):
        idx = rng.integers(0, n, n)
        xb = torch.tensor(x_np[idx], dtype=torch.float32)
        tb = torch.tensor(t_np[idx], dtype=torch.float32)
        yb = torch.tensor(y_np[idx], dtype=torch.float32)
        return fit_cevae_once(
            xb, tb, yb, feature_dim=feature_dim, outcome_dist=outcome_dist,
            hidden_dim=hidden_dim, num_layers=num_layers, num_epochs=num_epochs,
            seed=seed + i,
        )

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one_run)(i) for i in range(n_boot)
    )
    return np.array(results)


def default_n_jobs():
    return min(os.cpu_count() or 1, 8)
