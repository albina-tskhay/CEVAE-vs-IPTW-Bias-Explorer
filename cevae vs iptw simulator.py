import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import torch
import pyro
import pyro.distributions as dist
import os
from tqdm import tqdm
from joblib import Parallel, delayed

# ----------------- CONFIG ------------------

def simulate_cevae(x_train, t_train, y_train, x_test, args, num_bootstrap=50):
    from cevae_test_mod import CEVAE  # Make sure this file exists

    def balanced_bootstrap_resample(x, t, y, B):
        n = len(x)
        reps = B // n
        rem = B % n
        inds = torch.arange(n).repeat(reps).tolist()
        inds += torch.randint(0, n, (rem,)).tolist()
        np.random.shuffle(inds)
        return x[inds], t[inds], y[inds]

    def train_and_estimate_ate_balanced():
        xr, tr, yr = balanced_bootstrap_resample(x_train, t_train, y_train, len(x_train))
        pyro.clear_param_store()
        cevae = CEVAE(
            feature_dim=args['feature_dim'],
            latent_dim=1,
            hidden_dim=140,
            num_layers=3,
            num_samples=100,
        )
        cevae.fit(
            xr, tr, yr,
            num_epochs=200,
            batch_size=64,
            learning_rate=1e-4,
            learning_rate_decay=0.5,
            weight_decay=1e-5
        )

        with torch.no_grad():
            ite = cevae.ite(x_test).cpu().numpy()
        return ite.mean()

    n_jobs = min(num_bootstrap, os.cpu_count() or 1)
    ates = Parallel(n_jobs=n_jobs)(
        delayed(train_and_estimate_ate_balanced)() for _ in tqdm(range(num_bootstrap))
    )
    lower, upper = np.percentile(ates, [2.5, 97.5])
    return np.mean(ates), lower, upper

# ----------------- Streamlit App ------------------

st.set_page_config(page_title="CEVAE vs IPTW Bias Explorer", layout="centered")
st.title("CEVAE vs IPTW Bias Explorer")

st.markdown("""
Upload a dataset, choose Treatment (T), Outcome (Y), and Proxy variables.  
Then define the confounding structure:

- **β_T** controls the confounding strength of `U → T`
- **β_U** controls the confounding strength of `U → Y`
- **γ values** define how each proxy relates to the unobserved confounder `U`

We'll simulate data with latent confounding and estimate the treatment effect using:
**Naive**, **IPTW**, and **CEVAE**, comparing bias against a true effect of 0.05.
""")

uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head())

    # Variable selections
    TRUE_ATE = st.number_input("Specify the True ATE (used for bias calculation):", 
                           min_value=-1.0, max_value=1.0, value=0.05, step=0.01)
    t_var = st.selectbox("Select Treatment variable (T):", df.columns)
    y_var = st.selectbox("Select Outcome variable (Y):", df.columns)
    proxy_vars = st.multiselect("Select Proxy variable(s):", [col for col in df.columns if col not in [t_var, y_var]])

    # Confounding inputs
    beta_t = st.slider("β_T (U → T)", -2.0, 2.0, 0.8, 0.1)
    beta_u = st.slider("β_U (U → Y)", -2.0, 2.0, 0.5, 0.1)
    gamma_vals = st.text_input("γ values for proxies (comma-separated)", "0.3,0.3,0.3")

    # Simulation controls
    num_bootstrap = st.number_input("Number of Bootstrap Iterations", min_value=10, max_value=1000, value=50, step=10)
    sample_size = st.number_input("Subsample size for simulation", min_value=100, max_value=df.shape[0], value=min(1000, df.shape[0]), step=50)

    if st.button("🚀 Simulate + Estimate Effects"):
        gamma = np.array([float(g.strip()) for g in gamma_vals.split(",")])
        if len(gamma) != len(proxy_vars):
            st.error("❌ Length of γ values must match number of proxies.")
            st.stop()

        # Sample for speed
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        n = df.shape[0]
        U = np.random.normal(0, 1, size=n)

        Z_new = pd.DataFrame()
        for i, (zcol, g) in enumerate(zip(proxy_vars, gamma)):
            noise = np.random.normal(0, 1, size=n)
            Z_new[zcol + "_proxy"] = g * U + (1 - abs(g)) * noise

        logits_t = beta_t * U + np.random.normal(0, 1, size=n)
        T_sim = (1 / (1 + np.exp(-logits_t))) > 0.5
        T_sim = T_sim.astype(int)

        logits_y = beta_u * U + 0.5 * T_sim + np.random.normal(0, 1, size=n)
        Y_sim = (1 / (1 + np.exp(-logits_y))) > 0.5
        Y_sim = Y_sim.astype(int)

        df_sim = pd.DataFrame({"T": T_sim, "Y": Y_sim})
        df_sim = pd.concat([df_sim, Z_new], axis=1)

        # Check for NaN-safe ATE
        if df_sim["T"].nunique() < 2:
            naive_ate = np.nan
            naive_bias = np.nan
        else:
            treated = df_sim[df_sim["T"] == 1]["Y"]
            control = df_sim[df_sim["T"] == 0]["Y"]
            naive_ate = treated.mean() - control.mean()
            se_naive = np.sqrt(treated.var(ddof=1)/len(treated) + control.var(ddof=1)/len(control))
            ci_naive_low = naive_ate - 1.96 * se_naive
            ci_naive_high = naive_ate + 1.96 * se_naive
            naive_bias = 100 * (naive_ate - TRUE_ATE) / TRUE_ATE


        Z = df_sim[[col for col in df_sim.columns if col.endswith("_proxy")]]
        T = df_sim["T"]
        Y = df_sim["Y"]

        ps_model = LogisticRegression()
        ps_model.fit(Z, T)
        ps = ps_model.predict_proba(Z)[:, 1]
        weights = T / ps + (1 - T) / (1 - ps)
        iptw_model = sm.WLS(Y, sm.add_constant(T), weights=weights).fit()
        iptw_ate = iptw_model.params.iloc[1]
        se_iptw = iptw_model.bse.iloc[1]
        ci_iptw_low = iptw_ate - 1.96 * se_iptw
        ci_iptw_high = iptw_ate + 1.96 * se_iptw
        iptw_bias = 100 * (iptw_ate - TRUE_ATE) / TRUE_ATE
        iptw_bias = 100 * (iptw_ate - TRUE_ATE) / TRUE_ATE

        x_all = torch.tensor(Z.values, dtype=torch.float32)
        t_all = torch.tensor(T.values, dtype=torch.float32)
        y_all = torch.tensor(Y.values, dtype=torch.float32)

        ate_cevae, ci_lo, ci_hi = simulate_cevae(
            x_all, t_all, y_all, x_all,
            {'feature_dim': Z.shape[1]},
            num_bootstrap=num_bootstrap
        )
        cevae_bias = 100 * (ate_cevae - TRUE_ATE) / TRUE_ATE

        results_df = pd.DataFrame({
        "Method": ["Naive", "IPTW", "CEVAE"],
        "ATE Estimate": [naive_ate, iptw_ate, ate_cevae],
        "95% CI Lower": [ci_naive_low, ci_iptw_low, ci_lo],
        "95% CI Upper": [ci_naive_high, ci_iptw_high, ci_hi],
        "% Bias": [naive_bias, iptw_bias, cevae_bias]
        })


        st.subheader("📊 % Bias Results")
        if not results_df.empty:
            st.success("✅ Simulation complete.")
            st.subheader("📊 % Bias Results")
            st.dataframe(results_df.round(3))
        else:
            st.warning("⚠️ No results to display. Please check your inputs.")
        