"""Model visualization functions for Bayesian hierarchical models."""

import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

logger: logging.Logger = logging.getLogger(__name__)


def plot_shrinkage_regression(
    df: pd.DataFrame,
    idata: az.InferenceData,
    figsize: tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Plot unpooled OLS vs hierarchical regression lines per generation.

    Shows how the hierarchical model "shrinks" generation-specific regression lines
    toward the grand mean, especially for data-sparse generations.

    Args:
        df: Model-ready DataFrame with 'generation', 'age', 'log_price' columns
        idata: ArviZ InferenceData with posterior samples
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    gen_counts = df.groupby("generation").size().sort_values()
    generations_ordered = gen_counts.index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=False, sharey=True)
    axes = axes.flatten()

    grand_intercept = idata.posterior["Intercept"].mean().item()
    global_age_coef = idata.posterior["age"].mean().item()

    shrinkage_summary = []

    for idx, gen in enumerate(generations_ordered):
        ax = axes[idx]
        gen_data = df[df["generation"] == gen]
        n_obs = len(gen_data)

        ax.scatter(
            gen_data["age"], gen_data["log_price"], alpha=0.4, s=20, color="gray", label="Data"
        )

        X_ols = gen_data[["age"]].values
        y_ols = gen_data["log_price"].values
        ols = LinearRegression().fit(X_ols, y_ols)

        gen_effect = (
            idata.posterior["1|generation"].sel({"generation__factor_dim": gen}).mean().item()
        )
        hier_intercept = grand_intercept + gen_effect

        age_range = np.linspace(gen_data["age"].min(), gen_data["age"].max(), 50)

        ols_pred = ols.predict(age_range.reshape(-1, 1))
        ax.plot(age_range, ols_pred, "r--", lw=2, label="Unpooled OLS")

        hier_pred = hier_intercept + global_age_coef * age_range
        ax.plot(age_range, hier_pred, "b-", lw=2, label="Hierarchical")

        ax.set_title(f"{gen} (n={n_obs})", fontsize=10)
        ax.set_xlabel("Age (years)")
        if idx % 4 == 0:
            ax.set_ylabel("log(price)")

        shrinkage_summary.append(
            {
                "generation": gen,
                "n": n_obs,
                "ols_intercept": ols.intercept_,
                "hier_intercept": hier_intercept,
                "shrinkage": hier_intercept - ols.intercept_,
            }
        )

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Shrinkage on Regression Lines: Unpooled OLS (dashed) vs Hierarchical (solid)", y=1.02
    )
    fig.tight_layout()

    return fig, pd.DataFrame(shrinkage_summary)


def plot_prior_posterior(
    idata: az.InferenceData,
    params: dict[str, dict] | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> plt.Figure:
    """Plot prior vs posterior comparison ridgeplot.

    Args:
        idata: ArviZ InferenceData with posterior samples
        params: Dict mapping param names to {'prior': scipy.stats dist, 'label': str}.
                If None, uses default params for standard Bambi hierarchical model.
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    if params is None:
        params = {
            "1|generation_sigma": {"prior": stats.halfnorm(scale=0.5), "label": "Generation SD"},
            "1|trim_tier_sigma": {"prior": stats.halfnorm(scale=0.5), "label": "Trim Tier SD"},
            "1|trans_type_sigma": {"prior": stats.halfnorm(scale=0.3), "label": "Trans Type SD"},
            "age": {"prior": stats.norm(loc=0, scale=0.2539), "label": "Age Effect"},
            "mileage_scaled": {
                "prior": stats.norm(loc=0, scale=2.1789),
                "label": "Mileage Effect",
            },
            "is_low_mileage": {
                "prior": stats.norm(loc=0, scale=4.7624),
                "label": "Low Mileage Effect",
            },
        }

    fig, axes = plt.subplots(len(params), 1, figsize=figsize, sharex=False)

    for idx, (param_name, param_info) in enumerate(params.items()):
        ax = axes[idx]
        posterior_samples = idata.posterior[param_name].values.flatten()

        if "sigma" in param_name:
            x = np.linspace(0, max(1.2, np.percentile(posterior_samples, 99)), 200)
        else:
            margin = 3 * posterior_samples.std()
            x = np.linspace(posterior_samples.min() - margin, posterior_samples.max() + margin, 200)

        prior_pdf = param_info["prior"].pdf(x)
        prior_pdf = prior_pdf / (prior_pdf.max() + 1e-10)

        kde = gaussian_kde(posterior_samples)
        posterior_pdf = kde(x)
        posterior_pdf = posterior_pdf / posterior_pdf.max()

        ax.fill_between(x, prior_pdf, alpha=0.3, color="orange", label="Prior")
        ax.fill_between(x, posterior_pdf, alpha=0.6, color="steelblue", label="Posterior")
        ax.plot(x, prior_pdf, color="orange", lw=1.5, linestyle="--")
        ax.plot(x, posterior_pdf, color="steelblue", lw=1.5)
        ax.axvline(posterior_samples.mean(), color="steelblue", linestyle=":", alpha=0.8)

        ax.set_ylabel(param_info["label"], fontsize=10)
        ax.set_yticks([])

        post_mean = posterior_samples.mean()
        post_ci = np.percentile(posterior_samples, [5, 95])
        ax.text(
            0.98,
            0.8,
            f"μ={post_mean:.3f}\n90% CI: [{post_ci[0]:.3f}, {post_ci[1]:.3f}]",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Parameter Value")
    fig.suptitle("Prior vs Posterior: How Much Did the Data Update Our Beliefs?", y=1.01)
    fig.tight_layout()

    return fig
