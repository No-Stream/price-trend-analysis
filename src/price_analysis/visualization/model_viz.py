"""Model visualization functions for Bayesian hierarchical models."""

import logging
from collections.abc import Callable

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
        # Match priors from hierarchical.py DEFAULT_PRIORS
        # For truncated normals: truncnorm(a, b, loc, scale) where a,b are standardized bounds
        params = {
            "1|generation_sigma": {"prior": stats.halfnorm(scale=0.5), "label": "Generation SD"},
            "1|trim_tier_sigma": {"prior": stats.halfnorm(scale=0.7), "label": "Trim Tier SD"},
            "1|trans_type_sigma": {"prior": stats.halfnorm(scale=0.3), "label": "Trans Type SD"},
            # Truncated Normal(0, 0.05) bounded ≤0 (negative half-normal)
            "age": {
                "prior": stats.truncnorm(a=-np.inf, b=0, loc=0, scale=0.05),
                "label": "Age Effect",
            },
            # Truncated Normal(0, 0.3) bounded ≤0 (negative half-normal)
            "mileage_scaled": {
                "prior": stats.truncnorm(a=-np.inf, b=0, loc=0, scale=0.3),
                "label": "Mileage Effect",
            },
            # Truncated Normal(0, 0.2) bounded ≥0 (positive half-normal)
            "is_low_mileage": {
                "prior": stats.truncnorm(a=0, b=np.inf, loc=0, scale=0.2),
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


def plot_shrinkage_comparison(
    df: pd.DataFrame,
    idata: az.InferenceData,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Plot raw group means vs hierarchical estimates for multiple grouping variables.

    Shows how partial pooling shrinks estimates toward the grand mean, with larger
    shrinkage for smaller groups or more extreme values.

    Args:
        df: Model-ready DataFrame with 'generation', 'trim_tier', 'trans_type', 'log_price' columns
        idata: ArviZ InferenceData with posterior samples
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    grand_mean = idata.posterior["Intercept"].mean().item()

    group_configs = [
        ("generation", "generation__factor_dim", "Generation"),
        ("trim_tier", "trim_tier__factor_dim", "Trim Tier"),
        ("trans_type", "trans_type__factor_dim", "Transmission"),
    ]

    for ax, (col, dim_name, title) in zip(axes, group_configs, strict=True):
        raw_means = df.groupby(col)["log_price"].mean()
        counts = df.groupby(col).size()

        hier_estimates = {}
        for group in raw_means.index:
            effect = idata.posterior[f"1|{col}"].sel({dim_name: group}).mean().item()
            hier_estimates[group] = grand_mean + effect

        for group in raw_means.index:
            raw = raw_means[group]
            hier = hier_estimates[group]
            count = counts[group]
            ax.scatter(raw, hier, s=count / 2, alpha=0.7)
            ax.annotate(group, (raw, hier), fontsize=8, xytext=(5, 5), textcoords="offset points")

        lims = [
            min(raw_means.min(), min(hier_estimates.values())) - 0.1,
            max(raw_means.max(), max(hier_estimates.values())) + 0.1,
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="No pooling (1:1)")
        ax.axhline(grand_mean, color="blue", linestyle=":", alpha=0.5, label="Complete pooling")
        ax.set_xlabel("Raw Group Mean (log price)")
        ax.set_ylabel("Hierarchical Estimate (log price)")
        ax.set_title(f"{title}: Shrinkage Plot")
        ax.legend(fontsize=8)

    fig.suptitle("Shrinkage: Points above 1:1 line are shrunk toward grand mean", y=1.02)
    fig.tight_layout()

    return fig


def plot_depreciation_curves(
    predict_fn: Callable,
    idata: az.InferenceData,
    generations: list[str],
    ages: range | list[int],
    mileage_fn: Callable[[int], int],
    title: str,
    mileage_mean: float,
    mileage_std: float,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Plot predicted prices across generations as age varies.

    Args:
        predict_fn: Callable that takes generation, age, mileage, mileage_mean, mileage_std
                   and returns dict with 'price' containing 'median' and 'ci_80' keys
        idata: ArviZ InferenceData with posterior samples
        generations: List of generation strings to plot
        ages: Range or list of ages to predict
        mileage_fn: Callable that maps age to mileage (e.g., lambda age: 5000 * age)
        title: Plot title
        mileage_mean: Mean mileage used for scaling
        mileage_std: Std dev of mileage used for scaling
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for gen in generations:
        medians = []
        ci_lows = []
        ci_highs = []

        for age in ages:
            mileage = mileage_fn(age)
            pred = predict_fn(
                generation=gen,
                age=age,
                mileage=mileage,
                mileage_mean=mileage_mean,
                mileage_std=mileage_std,
            )
            medians.append(pred["price"]["median"])
            ci_lows.append(pred["price"]["ci_80"][0])
            ci_highs.append(pred["price"]["ci_80"][1])

        ages_list = list(ages)
        ax.plot(ages_list, [m / 1000 for m in medians], label=gen, marker="o", markersize=4)
        ax.fill_between(
            ages_list,
            [lo / 1000 for lo in ci_lows],
            [hi / 1000 for hi in ci_highs],
            alpha=0.15,
        )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Predicted Price ($k)")
    ax.set_title(title)
    ax.legend(title="Generation", loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_cost_per_mile(
    idata: az.InferenceData,
    mileage_std: float,
    car_values: list[int] | None = None,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot cost per mile analysis showing percentage and dollar depreciation.

    Two-panel figure showing:
    - Left: Percentage depreciation per 10k miles (constant across car values)
    - Right: Dollar depreciation per mile by car value

    Args:
        idata: ArviZ InferenceData with posterior samples
        mileage_std: Std dev of mileage used for scaling
        car_values: List of car values in dollars for dollar depreciation calc.
                   Defaults to [50000, 100000, 150000, 200000, 300000]
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    if car_values is None:
        car_values = [50000, 100000, 150000, 200000, 300000]

    mileage_coef_samples = idata.posterior["mileage_scaled"].values.flatten()
    pct_per_mile = (1 - np.exp(mileage_coef_samples / mileage_std)) * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    pct_per_10k = pct_per_mile * 10000
    ax.hist(pct_per_10k, bins=50, alpha=0.7, density=True, color="steelblue")
    ax.axvline(
        np.median(pct_per_10k),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(pct_per_10k):.1f}%",
    )
    ax.axvline(np.percentile(pct_per_10k, 5), color="orange", linestyle=":", label="90% CI")
    ax.axvline(np.percentile(pct_per_10k, 95), color="orange", linestyle=":")
    ax.set_xlabel("Depreciation per 10k miles (%)")
    ax.set_ylabel("Density")
    ax.set_title("Percentage Depreciation per 10k Miles")
    ax.legend()

    ax = axes[1]
    for val in car_values:
        dollar_per_mile = val * (1 - np.exp(mileage_coef_samples / mileage_std))
        ax.hist(dollar_per_mile, bins=50, alpha=0.5, density=True, label=f"${val / 1000:.0f}k car")

    ax.set_xlabel("Depreciation per mile ($)")
    ax.set_ylabel("Density")
    ax.set_title("Dollar Depreciation per Mile (varies by car value)")
    ax.legend()

    fig.tight_layout()

    return fig


def plot_dollar_premiums(
    effects_df: pd.DataFrame,
    reference_price: float,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot horizontal bar chart of dollar-denominated premiums.

    Converts log-scale effects to dollar premiums relative to a reference price.
    Dollar premium = reference_price * (exp(effect) - 1)

    Args:
        effects_df: DataFrame with columns 'group' and 'effect' (log-scale effect)
        reference_price: Reference car price in dollars
        title: Optional plot title. Defaults to showing reference price.
        figsize: Figure size tuple

    Returns:
        Matplotlib Figure object
    """
    effects_df = effects_df.copy()
    effects_df["dollar_premium"] = reference_price * (np.exp(effects_df["effect"]) - 1)
    effects_df = effects_df.sort_values("dollar_premium")

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["red" if x < 0 else "steelblue" for x in effects_df["dollar_premium"]]
    bars = ax.barh(
        effects_df["group"], effects_df["dollar_premium"] / 1000, color=colors, alpha=0.7
    )

    ax.axvline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Premium/Discount ($k)")
    ax.set_ylabel("Group")

    if title is None:
        title = f"Dollar Premiums (reference: ${reference_price / 1000:.0f}k car)"
    ax.set_title(title)

    for bar, premium in zip(bars, effects_df["dollar_premium"], strict=True):
        width = bar.get_width()
        label_x = width + 0.5 if width >= 0 else width - 0.5
        ha = "left" if width >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"${premium / 1000:+.1f}k",
            va="center",
            ha=ha,
            fontsize=9,
        )

    fig.tight_layout()

    return fig
