"""Spline-based Bayesian models for Porsche 911 price analysis.

Uses B-splines for flexible nonlinear effects on continuous predictors
(age, log-mileage) while retaining partial pooling (random intercepts)
on categorical predictors.
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
    "color_category_sd": 0.3,
}


def build_spline_model(
    df: pd.DataFrame,
    age_df: int = 6,
    mileage_df: int = 6,
    include_sale_year: bool = True,
    include_color: bool = False,
    priors: dict[str, Any] | None = None,
) -> bmb.Model:
    """Build spline model with partial pooling on categoricals.

    Model structure
    ---------------
    log(price) = α + f(age) + g(log_mileage) + β_year
                 + α_gen      [per generation]
                 + α_trim     [per trim_tier]
                 + α_trans    [per trans_type]
                 + α_body     [per body_style]
                 + α_color    [per color_category, optional]
                 + ε

    where f() and g() are B-spline basis expansions.

    Why log-mileage?
    ----------------
    Log transform captures diminishing marginal effect of additional miles:
    going from 5k→10k miles matters more than 50k→55k miles.
    Combined with splines, this eliminates the need for a separate
    is_low_mileage binary indicator.

    Why splines instead of linear?
    ------------------------------
    EDA shows nonlinear depreciation: steeper in early years, flattening later.
    Splines capture this flexibility while remaining interpretable.
    We keep random intercepts for partial pooling on categoricals.

    Args:
        df: Model-ready DataFrame with required columns
        age_df: Degrees of freedom for age spline (default 6)
        mileage_df: Degrees of freedom for log-mileage spline (default 6)
        include_sale_year: Whether to include sale_year as fixed effect
        include_color: Whether to include color_category as random intercept
        priors: Optional dict of custom priors

    Returns:
        Bambi Model object (unfitted)
    """

    required_cols = [
        "log_price",
        "age",
        "log_mileage",
        "generation",
        "trim_tier",
        "trans_type",
        "body_style",
    ]
    if include_sale_year:
        required_cols.append("sale_year")
    if include_color:
        required_cols.append("color_category")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    age_spline = f"bs(age, df={age_df})"
    mileage_spline = f"bs(log_mileage, df={mileage_df})"

    formula_parts = [
        "log_price ~ 1",
        age_spline,
        mileage_spline,
        "(1 | generation)",
        "(1 | trim_tier)",
        "(1 | trans_type)",
        "(1 | body_style)",
    ]

    if include_sale_year:
        formula_parts.insert(3, "sale_year")

    if include_color:
        formula_parts.append("(1 | color_category)")

    formula = " + ".join(formula_parts)
    logger.info(f"Spline model formula: {formula}")

    prior_config = {**DEFAULT_SPLINE_PRIORS, **(priors or {})}
    bambi_priors = _build_spline_priors(prior_config, include_color=include_color)

    model = bmb.Model(formula, data=df, priors=bambi_priors, family="gaussian")
    logger.info(f"Built spline model with {len(df)} observations")

    return model


def _build_spline_priors(config: dict[str, float], include_color: bool = False) -> dict[str, Any]:
    """Convert config dict to Bambi prior specifications for spline model."""
    priors = {
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

    if include_color:
        priors["1|color_category"] = bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["color_category_sd"])
        )

    return priors


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

    summary = az.summary(idata, var_names=["~1|"], filter_vars="like")
    logger.info(f"Model summary:\n{summary}")

    rhat_max = summary["r_hat"].max()
    if rhat_max > 1.01:
        logger.warning(f"Potential convergence issues: max R-hat = {rhat_max:.3f}")

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

    Note:
        This function modifies `idata` in-place by adding posterior
        predictive samples for the new data grid.

    Args:
        model: Fitted Bambi model
        idata: InferenceData from model fitting (modified in-place)
        df: Original data (for computing medians and ranges)
        variable: Variable to plot ("age" or "log_mileage_scaled")
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

    other_var = "log_mileage" if variable == "age" else "age"
    other_median = df[other_var].median()

    gen_mode = df["generation"].mode().iloc[0]
    trim_mode = df["trim_tier"].mode().iloc[0]
    trans_mode = df["trans_type"].mode().iloc[0]
    body_mode = df["body_style"].mode().iloc[0]

    new_data = pd.DataFrame(
        {
            variable: var_grid,
            other_var: other_median,
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

    if "color_category" in df.columns:
        color_mode = df["color_category"].mode().iloc[0]
        new_data["color_category"] = pd.Categorical(
            [color_mode] * n_points, categories=df["color_category"].cat.categories
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

    var_label = "Age (years)" if variable == "age" else "log(Mileage)"
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
    """Plot age and log-mileage spline effects side by side.

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
    plot_spline_effect(model, idata, df, "log_mileage", ax=axes[1])

    fig.tight_layout()
    return fig


def predict_spline_price(
    model: bmb.Model,
    idata: az.InferenceData,
    df: pd.DataFrame,
    generation: str,
    trim_tier: str,
    trans_type: str,
    body_style: str,
    model_year: int,
    mileage: int,
    sale_year: int,
    include_sale_year: bool = False,
    color_category: str | None = None,
) -> dict:
    """Predict price distribution for a specific car using the spline model.

    Unlike the hierarchical model's predict_price(), this function uses
    log_mileage instead of mileage_scaled + is_low_mileage.

    Args:
        model: Fitted Bambi spline model
        idata: InferenceData from model fitting
        df: Original data (for categorical levels)
        generation: Car generation (e.g., "992.1")
        trim_tier: Trim tier (e.g., "sport", "base", "gt", "turbo")
        trans_type: Transmission type (e.g., "manual", "pdk", "auto")
        body_style: Body style (e.g., "coupe", "cabriolet", "targa")
        model_year: Model year of car
        mileage: Current mileage (raw miles, will be log-transformed)
        sale_year: Year of (hypothetical) sale
        include_sale_year: Whether the model includes sale_year as predictor
        color_category: Color category (e.g., "standard", "special", "pts").
            If None, uses mode from df. Only needed if model was built with
            include_color=True.

    Returns:
        Dict with price predictions and uncertainty intervals, compatible
        with hierarchical model's predict_price() output structure.
    """
    age = sale_year - model_year
    log_mileage = np.log(max(mileage, 1))

    new_data = pd.DataFrame(
        {
            "age": [age],
            "log_mileage": [log_mileage],
            "generation": pd.Categorical([generation], categories=df["generation"].cat.categories),
            "trim_tier": pd.Categorical([trim_tier], categories=df["trim_tier"].cat.categories),
            "trans_type": pd.Categorical([trans_type], categories=df["trans_type"].cat.categories),
            "body_style": pd.Categorical([body_style], categories=df["body_style"].cat.categories),
        }
    )

    if include_sale_year:
        new_data["sale_year"] = sale_year

    if "color_category" in df.columns:
        color_val = color_category if color_category else df["color_category"].mode().iloc[0]
        new_data["color_category"] = pd.Categorical(
            [color_val], categories=df["color_category"].cat.categories
        )

    model.predict(idata, data=new_data, kind="response", inplace=True)

    log_price_samples = idata.posterior_predictive["log_price"].values.flatten()
    price_samples = np.exp(log_price_samples)

    return {
        "config": {
            "generation": generation,
            "trim_tier": trim_tier,
            "trans_type": trans_type,
            "body_style": body_style,
            "model_year": model_year,
            "mileage": mileage,
            "sale_year": sale_year,
            "age": age,
        },
        "log_price": {
            "mean": float(np.mean(log_price_samples)),
            "std": float(np.std(log_price_samples)),
        },
        "price": {
            "mean": float(np.mean(price_samples)),
            "median": float(np.median(price_samples)),
            "std": float(np.std(price_samples)),
            "ci_80": [
                float(np.percentile(price_samples, 10)),
                float(np.percentile(price_samples, 90)),
            ],
            "ci_95": [
                float(np.percentile(price_samples, 2.5)),
                float(np.percentile(price_samples, 97.5)),
            ],
        },
        "samples": price_samples,
    }
