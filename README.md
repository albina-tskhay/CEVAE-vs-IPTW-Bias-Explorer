# CEVAE-vs-IPTW-Bias-Explorer
This interactive web-based simulation tool allows users to compare naive, IPTW, and CEVAE (by Louizos et al. 2017 https://arxiv.org/abs/1705.08821) estimators under controlled confounding scenarios with latent variables.
Users can:

Upload a custom CSV dataset

Select treatment (T), outcome (Y), and observed proxy variables

Define confounding parameters:

β<sub>T</sub>: strength of confounder effect on treatment (U → T)

β<sub>U</sub>: strength of confounder effect on outcome (U → Y)

γ values: correlation of each proxy with the unobserved confounder U

Choose:

Sample size for simulation

Number of bootstrap iterations

True Average Treatment Effect (ATE)

The tool simulates data with latent confounding, computes treatment effects via:

Naive estimation (mean difference)

IPTW (inverse probability of treatment weighting)

CEVAE (causal effect variational autoencoder)

It reports the estimated ATEs, 95% confidence intervals, and % bias relative to the specified true ATE.
