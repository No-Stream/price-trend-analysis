"""Spline-based Bayesian models for Porsche 911 price analysis.

Uses B-splines or cubic regression splines for flexible nonlinear effects
on continuous predictors (age, mileage) while retaining partial pooling
(random intercepts) on categorical predictors.
"""

import logging
from typing import Any

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


DEFAULT_SPLINE_PRIORS = {
    "generation_sd": 0.5,
    "trim_sd": 0.7,
    "transmission_sd": 0.3,
    "body_style_sd": 0.3,
    "low_mileage_sigma": 0.2,
}


def build_spline_model(
    df: pd.DataFrame,
    age_df: int = 4,
    mileage_df: int = 4,
    include_sale_year: bool = True,
    priors: dict[str, Any] | None = None,
) -> bmb.Model:
    """Build spline model with partial pooling on categoricals.

    Model structure
    ---------------
    log(price) = α + f(age) + g(mileage) + β_low_mileage + β_year
                 + α_gen      [per generation]
                 + α_trim     [per trim_tier]
                 + α_trans    [per trans_type]
                 + α_body     [per body_style]
                 + ε

    where f() and g() are B-spline basis expansions.

    Why splines instead of linear?
    ------------------------------
    EDA shows nonlinear depreciation: steeper in early years, flattening later.
    Splines capture this flexibility while remaining interpretable.
    We keep random intercepts for partial pooling on categoricals.

    Args:
        df: Model-ready DataFrame with required columns
        age_df: Degrees of freedom for age spline (default 4)
        mileage_df: Degrees of freedom for mileage spline (default 4)
        include_sale_year: Whether to include sale_year as fixed effect
        priors: Optional dict of custom priors

    Returns:
        Bambi Model object (unfitted)
    """

    required_cols = [
        "log_price",
        "age",
        "mileage_scaled",
        "is_low_mileage",
        "generation",
        "trim_tier",
        "trans_type",
        "body_style",
    ]
    if include_sale_year:
        required_cols.append("sale_year")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    age_spline = f"bs(age, df={age_df})"
    mileage_spline = f"bs(mileage_scaled, df={mileage_df})"

    formula_parts = [
        "log_price ~ 1",
        age_spline,
        mileage_spline,
        "is_low_mileage",
        "(1 | generation)",
        "(1 | trim_tier)",
        "(1 | trans_type)",
        "(1 | body_style)",
    ]

    if include_sale_year:
        formula_parts.insert(4, "sale_year")

    formula = " + ".join(formula_parts)
    logger.info(f"Spline model formula: {formula}")

    prior_config = {**DEFAULT_SPLINE_PRIORS, **(priors or {})}
    bambi_priors = _build_spline_priors(prior_config)

    model = bmb.Model(formula, data=df, priors=bambi_priors, family="gaussian")
    logger.info(f"Built spline model with {len(df)} observations")

    return model


def _build_spline_priors(config: dict[str, float]) -> dict[str, Any]:
    """Convert config dict to Bambi prior specifications for spline model."""
    return {
        "1|generation": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["generation_sd"])
        ),
        "1|trim_tier": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["trim_sd"])
        ),
        "1|trans_type": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["transmission_sd"])
        ),
        "1|body_style": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["body_style_sd"])
        ),
    }


def fit_spline_model(
    model: bmb.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 8,
    target_accept: float = 0.9,
    random_seed: int = 42,
    include_log_likelihood: bool = True,
    **kwargs,
) -> az.InferenceData:
    """Fit spline model using MCMC sampling.

    Args:
        model: Bambi Model object
        draws: Number of posterior samples per chain
        tune: Number of tuning/warmup samples
        chains: Number of MCMC chains
        target_accept: Target acceptance rate for NUTS sampler
        random_seed: Random seed for reproducibility
        include_log_likelihood: Whether to compute pointwise log-likelihood for LOO-CV
        **kwargs: Additional arguments passed to model.fit()

    Returns:
        ArviZ InferenceData with posterior samples and diagnostics
    """
    logger.info(f"Fitting spline model: {draws} draws, {tune} tune, {chains} chains")

    idata_kwargs = kwargs.pop("idata_kwargs", {})
    if include_log_likelihood:
        idata_kwargs["log_likelihood"] = True

    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        idata_kwargs=idata_kwargs,
        **kwargs,
    )

    return idata


def plot_spline_effect(
    model: bmb.Model,
    idata: az.InferenceData,
    df: pd.DataFrame,
    variable: str,
    n_points: int = 100,
    ax: plt.Axes | None = None,
    credible_interval: float = 0.9,
) -> plt.Axes:
    """Plot estimated spline curve with credible bands.

    Creates a grid of values for the target variable, holds other
    continuous variables at their median, and plots the predicted
    effect with uncertainty.

    Args:
        model: Fitted Bambi model
        idata: InferenceData from model fitting
        df: Original data (for computing medians and ranges)
        variable: Variable to plot ("age" or "mileage_scaled")
        n_points: Number of points in the grid
        ax: Optional axes to plot on
        credible_interval: Width of credible interval (default 0.9)

    Returns:
        Matplotlib Axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    var_min = df[variable].min()
    var_max = df[variable].max()
    var_grid = np.linspace(var_min, var_max, n_points)

    other_var = "mileage_scaled" if variable == "age" else "age"
    other_median = df[other_var].median()

    gen_mode = df["generation"].mode().iloc[0]
    trim_mode = df["trim_tier"].mode().iloc[0]
    trans_mode = df["trans_type"].mode().iloc[0]
    body_mode = df["body_style"].mode().iloc[0]

    new_data = pd.DataFrame(
        {
            variable: var_grid,
            other_var: other_median,
            "is_low_mileage": 0,
            "sale_year": df["sale_year"].median() if "sale_year" in df.columns else 2025,
            "generation": pd.Categorical(
                [gen_mode] * n_points, categories=df["generation"].cat.categories
            ),
            "trim_tier": pd.Categorical(
                [trim_mode] * n_points, categories=df["trim_tier"].cat.categories
            ),
            "trans_type": pd.Categorical(
                [trans_mode] * n_points, categories=df["trans_type"].cat.categories
            ),
            "body_style": pd.Categorical(
                [body_mode] * n_points, categories=df["body_style"].cat.categories
            ),
        }
    )

    model.predict(idata, data=new_data, kind="response", inplace=True)

    log_price_samples = idata.posterior_predictive["log_price"].values
    log_price_samples = log_price_samples.reshape(-1, n_points)

    mean_pred = log_price_samples.mean(axis=0)
    alpha = (1 - credible_interval) / 2
    lower = np.percentile(log_price_samples, alpha * 100, axis=0)
    upper = np.percentile(log_price_samples, (1 - alpha) * 100, axis=0)

    ax.plot(var_grid, mean_pred, "b-", linewidth=2, label="Posterior mean")
    ax.fill_between(
        var_grid, lower, upper, alpha=0.3, color="blue", label=f"{int(credible_interval * 100)}% CI"
    )

    ax.scatter(df[variable], df["log_price"], alpha=0.3, s=10, c="gray", label="Data")

    var_label = "Age (years)" if variable == "age" else "Mileage (z-scored)"
    ax.set_xlabel(var_label)
    ax.set_ylabel("log(price)")
    ax.set_title(f"Spline Effect: {variable}")
    ax.legend()

    return ax


def plot_spline_effects_grid(
    model: bmb.Model,
    idata: az.InferenceData,
    df: pd.DataFrame,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot age and mileage spline effects side by side.

    Args:
        model: Fitted Bambi model
        idata: InferenceData from model fitting
        df: Original data
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_spline_effect(model, idata, df, "age", ax=axes[0])
    plot_spline_effect(model, idata, df, "mileage_scaled", ax=axes[1])

    fig.tight_layout()
    return fig
