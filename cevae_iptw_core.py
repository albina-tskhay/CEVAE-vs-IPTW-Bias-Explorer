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
    """Fit IPTW via logistic-regression propensity scores. Returns (ate, (ci_low, ci_high)).
    Accepts t and y as either pandas Series or numpy arrays; statsmodels returns
    .params as a pandas Series for Series input but as a plain ndarray for
    array input, so we index with np.asarray(...)[1] rather than .iloc[1] to
    handle both cases."""
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(proxies_df, t)
    ps = np.clip(ps_model.predict_proba(proxies_df)[:, 1], 0.02, 0.98)
    t_arr = np.asarray(t)
    y_arr = np.asarray(y)
    weights = t_arr / ps + (1 - t_arr) / (1 - ps)
    model = sm.WLS(y_arr, sm.add_constant(t_arr), weights=weights).fit()
    ate = float(np.asarray(model.params)[1])
    se = float(np.asarray(model.bse)[1])
    return ate, (ate - 1.96 * se, ate + 1.96 * se)


def bootstrap_iptw(x_np, t_np, y_np, n_boot, n_jobs=1, seed=0, progress_callback=None):
    """Bootstrap CI for IPTW. Runs sequentially: each IPTW fit (logistic
    regression + weighted least squares) is fast enough on its own that
    process-based parallelism is not worth the overhead and risk here.
    (Process-based parallelism via joblib was tried for this and for the
    CEVAE bootstrap below; it hung indefinitely in constrained/containerized
    environments such as Streamlit Cloud, likely due to multiprocessing
    start-method restrictions in that environment, so it has been removed
    in favor of a sequential loop that is slower but reliably correct.)
    n_jobs is accepted but ignored, kept only so existing call sites do not
    need to change their argument list."""
    n = len(x_np)
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(x_np[idx], t_np[idx])
        ps = np.clip(ps_model.predict_proba(x_np[idx])[:, 1], 0.02, 0.98)
        w = t_np[idx] / ps + (1 - t_np[idx]) / (1 - ps)
        m = sm.WLS(y_np[idx], sm.add_constant(t_np[idx]), weights=w).fit()
        results.append(float(np.asarray(m.params)[1]))
        if progress_callback is not None:
            progress_callback(i + 1, n_boot)
    return np.array(results)


def bootstrap_cevae(x_np, t_np, y_np, outcome_dist, n_boot, n_jobs=1, seed=0,
                     hidden_dim=DEFAULT_HIDDEN_DIM, num_layers=DEFAULT_NUM_LAYERS,
                     num_epochs=DEFAULT_NUM_EPOCHS, progress_callback=None):
    """Bootstrap CI for CEVAE: refits the full model on each resample,
    sequentially. An earlier version of this function used joblib's
    process-based parallelism to run refits concurrently; that hung
    indefinitely in constrained/containerized environments such as
    Streamlit Cloud (most likely due to multiprocessing start-method
    restrictions there), so it was removed. This is now a plain sequential
    loop: slower, but reliable. Each CEVAE refit is the real cost in this
    bootstrap (seconds per fit), so n_boot directly controls wall-clock time;
    keep it modest (e.g. 20-50) for an interactive tool.
    n_jobs is accepted but ignored, kept only so existing call sites do not
    need to change their argument list."""
    n = len(x_np)
    rng = np.random.default_rng(seed)
    feature_dim = x_np.shape[1]
    results = []
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = torch.tensor(x_np[idx], dtype=torch.float32)
        tb = torch.tensor(t_np[idx], dtype=torch.float32)
        yb = torch.tensor(y_np[idx], dtype=torch.float32)
        results.append(fit_cevae_once(
            xb, tb, yb, feature_dim=feature_dim, outcome_dist=outcome_dist,
            hidden_dim=hidden_dim, num_layers=num_layers, num_epochs=num_epochs,
            seed=seed + i,
        ))
        if progress_callback is not None:
            progress_callback(i + 1, n_boot)
    return np.array(results)


def select_hyperparameters(x_np, t_np, y_np, outcome_dist, seed=0,
                            hidden_dim_options=(10, 20, 40),
                            num_layers_options=(2, 3),
                            num_epochs=60, val_frac=0.2):
    """Small validation-based hyperparameter search.

    Splits the data into train/validation, fits CEVAE under each
    (hidden_dim, num_layers) combination in the grid on the training split,
    scores each by held-out ELBO on the validation split (the same
    objective CEVAE is trained on; see Louizos et al. eq. 6), and returns
    the configuration with the best (highest) validation ELBO.

    This selection is run once on the full dataset to choose a single
    configuration; that configuration is then reused for the final point
    estimate and for every bootstrap resample (bootstrap does NOT repeat
    hyperparameter selection on every resample, which would multiply
    runtime by the grid size on top of the bootstrap count).

    Note: num_layers must be >= 2 for this CEVAE implementation; the
    internal network builder (FullyConnected in cevae_test_mod.py) raises
    IndexError on num_layers=1 (it builds an empty layer list once the
    final activation is removed). This was found by testing, not
    documented in the original file, so do not lower the default below 2.

    Returns (best_hidden_dim, best_num_layers, scores_dict) where
    scores_dict maps (hidden_dim, num_layers) -> validation ELBO, so the
    caller can show the user what was compared.
    """
    n = len(x_np)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 20)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    feature_dim = x_np.shape[1]
    x_train = torch.tensor(x_np[train_idx], dtype=torch.float32)
    t_train = torch.tensor(t_np[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_np[train_idx], dtype=torch.float32)
    x_val = torch.tensor(x_np[val_idx], dtype=torch.float32)
    t_val = torch.tensor(t_np[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_np[val_idx], dtype=torch.float32)

    scores = {}
    for hidden_dim in hidden_dim_options:
        for num_layers in num_layers_options:
            torch.manual_seed(seed)
            pyro.set_rng_seed(seed)
            pyro.clear_param_store()
            cevae = CEVAE(
                feature_dim=feature_dim, outcome_dist=outcome_dist, latent_dim=1,
                hidden_dim=hidden_dim, num_layers=num_layers, num_samples=100,
            )
            cevae.fit(
                x_train, t_train, y_train,
                num_epochs=num_epochs, batch_size=min(200, len(x_train)),
                learning_rate=1e-3, learning_rate_decay=0.5, weight_decay=1e-4,
                log_every=0,
            )
            # Held-out loss, using the same TraceCausalEffect_ELBO objective
            # CEVAE is actually trained with (not the generic Trace_ELBO,
            # which would evaluate a subtly different objective). cevae.fit()
            # whitens x internally using a whitener fit on the training
            # split; that same whitener (now stored as cevae.whiten) must be
            # applied to x_val too, or validation loss is computed on a
            # different input distribution than the model was trained on,
            # which would make every configuration look uniformly bad.
            from cevae_test_mod import TraceCausalEffect_ELBO
            elbo = TraceCausalEffect_ELBO()
            with torch.no_grad():
                x_val_whitened = cevae.whiten(x_val)
                val_loss = elbo.loss(cevae.model, cevae.guide, x_val_whitened, t_val, y_val, size=len(x_val))
            scores[(hidden_dim, num_layers)] = -val_loss  # higher (less negative loss) is better

    best_config = max(scores, key=scores.get)
    return best_config[0], best_config[1], scores
