"""
CEVAE vs IPTW Sensitivity Check
--------------------------------
Redesigned app:
- Mode toggle: "Use my data" (real T/Y/proxies) vs "Simulation" (teaching mode,
  synthetic U/T/Y from user-set sliders, for understanding method behavior
  under known ground truth).
- CI toggle: off (fast, point estimates only) vs on (bootstrapped CIs,
  slower, parallelized).
- No fixed divergence threshold. The app reports both estimates and the
  gap between them; the user decides what gap is meaningful for their case.
- Fast defaults (hidden_dim=20, num_layers=2, early stopping) so a single
  fit completes in well under a minute on a typical uploaded dataset,
  instead of the original ~50s-per-fit / 50-fit bootstrap design.
"""
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
import logging
logging.basicConfig(level=logging.WARNING)  # was DEBUG; that alone was slowing every run via console I/O

from cevae_iptw_core import (
    fit_cevae_once, fit_iptw, bootstrap_iptw, bootstrap_cevae, default_n_jobs,
    DEFAULT_HIDDEN_DIM, DEFAULT_NUM_LAYERS, DEFAULT_NUM_EPOCHS,
)


# ---------------- Streamlit App ------------------

st.set_page_config(page_title="CEVAE vs IPTW Sensitivity Check", layout="centered")
st.title("CEVAE vs IPTW Sensitivity Check")

st.markdown("""
This tool fits two methods, **IPTW** and **CEVAE**, to the same treatment,
outcome, and proxy variables, and reports both estimates side by side.

It does not tell you which estimate is correct. A close match between the
two is reassuring; a large gap suggests your result may be sensitive to the
choice of adjustment method, and to whether your proxies adequately capture
an unmeasured confounder. How large a gap matters is a judgment call for
you to make in the context of your own study.
""")

mode = st.radio(
    "Mode",
    ["Use my data", "Simulation (teaching mode)"],
    help="'Use my data' fits both methods on your actual treatment and outcome columns. "
         "'Simulation' generates a synthetic treatment and outcome from sliders you set, "
         "so you can see how each method behaves when the true effect is known. "
         "Simulation mode does NOT analyze your real treatment/outcome columns."
)

compute_ci = st.checkbox(
    "Compute confidence intervals (bootstrap)",
    value=False,
    help="Off: point estimates only, fast (one CEVAE fit). "
         "On: bootstrapped 95% CIs for both methods, slower (refits the model many times)."
)

uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded.")
    st.dataframe(df.head())

    if mode == "Use my data":
        t_var = st.selectbox("Treatment variable (T):", df.columns)
        y_var = st.selectbox("Outcome variable (Y):", df.columns)
        proxy_vars = st.multiselect(
            "Proxy variable(s) for the suspected unmeasured confounder:",
            [c for c in df.columns if c not in [t_var, y_var]]
        )

        sample_size = st.number_input(
            "Subsample size (smaller = faster; large datasets will be subsampled for CEVAE fitting)",
            min_value=100, max_value=df.shape[0], value=min(3000, df.shape[0]), step=100
        )

        if compute_ci:
            n_boot = st.number_input("Bootstrap iterations", min_value=20, max_value=500, value=100, step=20)

        run = st.button("Run comparison")

        if run:
            if not proxy_vars:
                st.error("Select at least one proxy variable.")
                st.stop()

            work = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            t = work[t_var].astype(float).values
            y = work[y_var].astype(float).values
            proxies_df = work[proxy_vars].astype(float)
            x_np = proxies_df.values
            outcome_dist = "bernoulli" if set(np.unique(y)) <= {0.0, 1.0} else "normal"

            n_jobs = default_n_jobs()

            t0 = time.time()
            with st.spinner("Fitting IPTW..."):
                iptw_ate, iptw_ci = fit_iptw(proxies_df, work[t_var], work[y_var])

            with st.spinner("Fitting CEVAE (single fit, fast defaults)..."):
                x_t = torch.tensor(x_np, dtype=torch.float32)
                t_t = torch.tensor(t, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                cevae_ate = fit_cevae_once(
                    x_t, t_t, y_t,
                    feature_dim=x_np.shape[1], outcome_dist=outcome_dist,
                    hidden_dim=20, num_layers=2,
                    num_epochs=80, batch_size=min(200, len(work)),
                    learning_rate=1e-3, seed=42,
                )
            fit_time = time.time() - t0

            cevae_ci = (np.nan, np.nan)
            iptw_ci_boot = iptw_ci
            if compute_ci:
                with st.spinner(f"Bootstrapping {n_boot} iterations for both methods (parallelized across {n_jobs} workers)..."):
                    iptw_boot = bootstrap_iptw(x_np, t, y, n_boot=n_boot, n_jobs=n_jobs)
                    iptw_ci_boot = tuple(np.percentile(iptw_boot, [2.5, 97.5]))

                    cevae_boot = bootstrap_cevae(x_np, t, y, outcome_dist, n_boot=n_boot, n_jobs=n_jobs)
                    cevae_ci = tuple(np.percentile(cevae_boot, [2.5, 97.5]))

            st.subheader("Results")
            results_df = pd.DataFrame({
                "Method": ["IPTW", "CEVAE"],
                "Estimate": [iptw_ate, cevae_ate],
                "95% CI Lower": [iptw_ci_boot[0], cevae_ci[0]],
                "95% CI Upper": [iptw_ci_boot[1], cevae_ci[1]],
            })
            st.dataframe(results_df.round(4))

            abs_diff = abs(iptw_ate - cevae_ate)
            pct_diff = 100 * abs_diff / abs(iptw_ate) if iptw_ate != 0 else float("nan")
            st.markdown(f"**Absolute difference between methods:** {abs_diff:.4f}")
            st.markdown(f"**Relative difference (vs. IPTW estimate):** {pct_diff:.1f}%")
            st.caption(
                "This tool does not flag a 'too large' difference automatically. "
                "Consider this gap in light of your proxies' likely adequacy, your "
                "sample size, and the clinical importance of the effect size in question."
            )
            st.caption(f"Total computation time: {fit_time:.1f} seconds (single-fit mode)" +
                       (f" + bootstrap" if compute_ci else ""))

    else:  # Simulation (teaching mode)
        st.info(
            "Simulation mode generates a synthetic latent confounder, treatment, "
            "and outcome from the sliders below. Your real treatment and outcome "
            "columns are NOT used in this mode; only the selected proxy columns "
            "are reused as templates for proxy noise structure. Use this mode to "
            "see how CEVAE and IPTW behave under known ground truth, not to "
            "analyze your own causal question."
        )
        proxy_vars = st.multiselect("Columns to use as proxy templates:", df.columns)
        true_ate = st.number_input("True ATE to simulate:", min_value=-1.0, max_value=1.0, value=0.05, step=0.01)
        beta_t = st.slider("Confounding strength: U -> T", -2.0, 2.0, 0.8, 0.1)
        beta_u = st.slider("Confounding strength: U -> Y", -2.0, 2.0, 0.5, 0.1)
        sample_size = st.number_input("Simulated sample size:", min_value=200, max_value=10000, value=2000, step=200)

        if st.button("Run simulation"):
            if not proxy_vars:
                st.error("Select at least one column to use as a proxy template.")
                st.stop()
            n = sample_size
            U = np.random.binomial(1, 0.5, size=n)
            Z = np.zeros((n, len(proxy_vars)))
            for j in range(len(proxy_vars)):
                Z[:, j] = 0.6*(2*U-1) + np.random.normal(0, 1, size=n)
            logit_t = beta_t * U + np.random.normal(0, 0.3, size=n)
            T_sim = (1/(1+np.exp(-logit_t)) > 0.5).astype(int)
            logit_y = -1.5 + true_ate*4*T_sim + beta_u*U  # scaled so true_ate maps to ~risk difference
            p_y = 1/(1+np.exp(-logit_y))
            Y_sim = np.random.binomial(1, p_y)

            iptw_ate, iptw_ci = fit_iptw(pd.DataFrame(Z), T_sim, Y_sim)
            x_t = torch.tensor(Z, dtype=torch.float32)
            t_t = torch.tensor(T_sim, dtype=torch.float32)
            y_t = torch.tensor(Y_sim, dtype=torch.float32)
            with st.spinner("Fitting CEVAE..."):
                cevae_ate = fit_cevae_once(
                    x_t, t_t, y_t, feature_dim=Z.shape[1], outcome_dist="bernoulli",
                    hidden_dim=20, num_layers=2, num_epochs=80,
                    batch_size=min(200, n), learning_rate=1e-3, seed=42,
                )

            st.subheader("Results (synthetic ground truth)")
            res = pd.DataFrame({
                "Method": ["True ATE", "IPTW", "CEVAE"],
                "Estimate": [true_ate, iptw_ate, cevae_ate],
            })
            st.dataframe(res.round(4))
else:
    st.info("Upload a CSV to begin.")
