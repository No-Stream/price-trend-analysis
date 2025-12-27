"""Model comparison utilities for Bayesian price models.

Provides LOO-CV comparison and residual diagnostics for comparing
different model specifications (e.g., linear vs spline).
"""

import logging

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


def compare_models_loo(
    idata_dict: dict[str, az.InferenceData],
    ic: str = "loo",
) -> pd.DataFrame:
    """Compare models using LOO-CV (or WAIC).

    Uses ArviZ's compare() function which computes:
    - ELPD (expected log pointwise predictive density)
    - SE of ELPD
    - ELPD difference from best model
    - Weights (stacking or pseudo-BMA)

    Args:
        idata_dict: Dict mapping model names to InferenceData objects.
            Each idata must have log_likelihood computed.
        ic: Information criterion - "loo" (default) or "waic"

    Returns:
        DataFrame with comparison results, sorted by ELPD (best first)

    Example:
        >>> comparison = compare_models_loo({
        ...     "hierarchical": idata_hierarchical,
        ...     "spline": idata_spline,
        ... })
        >>> display(comparison)
    """
    for name, idata in idata_dict.items():
        if "log_likelihood" not in idata.groups():
            raise ValueError(
                f"Model '{name}' missing log_likelihood. Refit with include_log_likelihood=True."
            )

    comparison = az.compare(idata_dict, ic=ic)
    logger.info(f"Model comparison (ic={ic}):\n{comparison}")

    return comparison


def get_residuals(
    model: bmb.Model,
    idata: az.InferenceData,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract residuals (observed - predicted) with posterior statistics.

    Args:
        model: Fitted Bambi model
        idata: InferenceData from model fitting
        df: Original data with log_price column

    Returns:
        DataFrame with columns:
        - observed: actual log_price
        - predicted_mean: posterior mean prediction
        - predicted_median: posterior median prediction
        - residual_mean: observed - predicted_mean
        - residual_median: observed - predicted_median
        - age, mileage_scaled: for diagnostic plots
    """
    model.predict(idata, data=df, kind="response", inplace=True)

    log_price_samples = idata.posterior_predictive["log_price"].values
    log_price_samples = log_price_samples.reshape(-1, len(df))

    predicted_mean = log_price_samples.mean(axis=0)
    predicted_median = np.median(log_price_samples, axis=0)

    result = pd.DataFrame(
        {
            "observed": df["log_price"].values,
            "predicted_mean": predicted_mean,
            "predicted_median": predicted_median,
            "residual_mean": df["log_price"].values - predicted_mean,
            "residual_median": df["log_price"].values - predicted_median,
            "age": df["age"].values,
            "mileage_scaled": df["mileage_scaled"].values,
        }
    )

    return result


def plot_residuals(
    residuals: pd.DataFrame,
    ax: plt.Axes | None = None,
    vs: str = "fitted",
) -> plt.Axes:
    """Plot residuals vs fitted values or a predictor.

    Args:
        residuals: DataFrame from get_residuals()
        ax: Optional axes to plot on
        vs: What to plot against - "fitted", "age", or "mileage_scaled"

    Returns:
        Matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if vs == "fitted":
        x = residuals["predicted_mean"]
        xlabel = "Fitted (posterior mean)"
    elif vs in ("age", "mileage_scaled"):
        x = residuals[vs]
        xlabel = "Age (years)" if vs == "age" else "Mileage (z-scored)"
    else:
        raise ValueError(f"vs must be 'fitted', 'age', or 'mileage_scaled', got '{vs}'")

    y = residuals["residual_mean"]

    ax.scatter(x, y, alpha=0.3, s=10)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    ax.axhline(y.mean(), color="blue", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residual (observed - fitted)")
    ax.set_title(f"Residuals vs {vs}")

    return ax


def plot_residual_comparison(
    residuals_dict: dict[str, pd.DataFrame],
    vs: str = "fitted",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Side-by-side residual plots for multiple models.

    Args:
        residuals_dict: Dict mapping model names to residual DataFrames
        vs: What to plot against - "fitted", "age", or "mileage_scaled"
        figsize: Optional figure size

    Returns:
        Matplotlib Figure
    """
    n_models = len(residuals_dict)
    if figsize is None:
        figsize = (6 * n_models, 5)

    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, residuals) in zip(axes, residuals_dict.items(), strict=True):
        plot_residuals(residuals, ax=ax, vs=vs)
        ax.set_title(f"{name}: Residuals vs {vs}")

    fig.tight_layout()
    return fig


def plot_residual_diagnostics(
    residuals_dict: dict[str, pd.DataFrame],
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Comprehensive residual diagnostics for model comparison.

    Creates a 3x2 grid:
    - Row 1: Residuals vs fitted (one per model)
    - Row 2: Residuals vs age (one per model)
    - Row 3: Residuals vs mileage (one per model)

    Args:
        residuals_dict: Dict mapping model names to residual DataFrames
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if len(residuals_dict) != 2:
        raise ValueError("plot_residual_diagnostics expects exactly 2 models")

    model_names = list(residuals_dict.keys())

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    for col, name in enumerate(model_names):
        residuals = residuals_dict[name]
        plot_residuals(residuals, ax=axes[0, col], vs="fitted")
        plot_residuals(residuals, ax=axes[1, col], vs="age")
        plot_residuals(residuals, ax=axes[2, col], vs="mileage_scaled")

        for row in range(3):
            axes[row, col].set_title(f"{name}: {axes[row, col].get_title().split(': ')[1]}")

    fig.tight_layout()
    return fig


def compute_residual_stats(residuals: pd.DataFrame) -> dict[str, float]:
    """Compute summary statistics for residuals.

    Args:
        residuals: DataFrame from get_residuals()

    Returns:
        Dict with RMSE, MAE, and other statistics
    """
    r = residuals["residual_mean"]

    return {
        "rmse": np.sqrt((r**2).mean()),
        "mae": np.abs(r).mean(),
        "mean": r.mean(),
        "std": r.std(),
        "median": r.median(),
        "iqr": r.quantile(0.75) - r.quantile(0.25),
    }


def compare_residual_stats(
    residuals_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare residual statistics across models.

    Args:
        residuals_dict: Dict mapping model names to residual DataFrames

    Returns:
        DataFrame with one row per model, columns for each statistic
    """
    stats = {name: compute_residual_stats(r) for name, r in residuals_dict.items()}
    return pd.DataFrame(stats).T
