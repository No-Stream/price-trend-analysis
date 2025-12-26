"""Bayesian hierarchical models for Porsche 911 price analysis.

Uses Bambi (built on PyMC) for accessible Bayesian regression with
hierarchical/multilevel structure.
"""

import logging
from typing import Any

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

# Default weakly informative priors for random effect SDs
# These constrain variance parameters to reasonable ranges while allowing data to dominate
# sigma=0.5 on log-price means ~65% price difference at 1 SD
DEFAULT_RE_PRIORS = {
    "generation_sd": 0.5,
    "trim_sd": 0.5,
    "transmission_sd": 0.3,
    "age_slope_sd": 0.1,  # depreciation rate variation across generations
}


def build_model(
    df: pd.DataFrame,
    include_color: bool = False,
    include_sale_year: bool = True,
    include_generation_slope: bool = True,
    use_trim_tier: bool = False,
    use_trans_type: bool = False,
    priors: dict[str, Any] | None = None,
) -> bmb.Model:
    """Build hierarchical Bayesian model for 911 prices.

    Model structure
    ---------------
    log(price) = α + β_age * age + β_mileage * mileage + β_year * sale_year
                 + (α_gen + β_age_gen * age)  [per generation, slope optional]
                 + α_trim                      [per trim or trim_tier]
                 + α_trans                     [per transmission]
                 + ε

    Random effects intuition
    ------------------------
    We use CROSSED (not nested) random effects for generation, trim, and
    transmission. This assumes:

    1. Generation random intercepts + slopes on age:
       - Different generations have different base prices (intercept)
       - Different generations depreciate at different rates (slope on age)
       - e.g., 992.1 might depreciate faster than 997.2 which might appreciate
       - Set include_generation_slope=False for sparse data

    2. Trim random intercepts only:
       - The "Carrera 4S premium" over base Carrera is relatively stable
       - Crossed effect lets us learn a shared "Turbo premium" that pools
         information across all generations
       - We assume the trim premium doesn't change much over time (no slope)
       - Set use_trim_tier=True to collapse trims into 4 performance tiers

    3. Transmission random intercepts only:
       - The "manual premium" is fairly consistent across models
       - Crossed effect shares info: manual premium on 991.2 informs
         estimate for manual premium on 992.1

    Why crossed (not nested)?
    -------------------------
    Nested effects (e.g., trim within generation) would estimate a separate
    "Carrera 4S effect" for each generation, with no information sharing.
    Crossed effects are more parsimonious - we estimate one "Carrera 4S effect"
    that applies across generations, while generation handles base price
    differences.

    If residual analysis suggests strong interactions (e.g., GT3 manual premium
    is very different from Carrera manual premium), we can add interaction terms.

    Why log(price)?
    ---------------
    - Prices are right-skewed; log stabilizes variance
    - Effects become multiplicative (more natural: "manuals sell for 15% more")
    - Coefficients interpretable as approximate percentage changes

    Args:
        df: Model-ready DataFrame with required columns
        include_color: Whether to include color_category as predictor
        include_sale_year: Whether to include sale_year as fixed effect.
            Set to False when all listings are from same year (no variance).
        include_generation_slope: Whether to include random slope on age for
            generation. Set to False for sparse data (need ~20+ obs per gen).
        use_trim_tier: Whether to use grouped trim_tier instead of individual
            trim levels. Requires prepare_model_data(group_trims=True).
        use_trans_type: Whether to use grouped trans_type (manual/pdk/auto)
            instead of 4 transmission levels. Requires prepare_model_data(group_trans=True).
        priors: Optional dict of custom priors. Keys can include:
            - 'generation_sd': HalfNormal sigma for generation intercept SD
            - 'trim_sd': HalfNormal sigma for trim intercept SD
            - 'transmission_sd': HalfNormal sigma for transmission intercept SD
            - 'age_slope_sd': HalfNormal sigma for generation age slope SD

    Returns:
        Bambi Model object (unfitted)
    """
    trim_col = "trim_tier" if use_trim_tier else "trim"
    trans_col = "trans_type" if use_trans_type else "transmission"

    required_cols = [
        "log_price",
        "age",
        "mileage_scaled",
        "generation",
        trim_col,
        trans_col,
    ]
    if include_sale_year:
        required_cols.append("sale_year")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build formula
    generation_re = "(1 + age | generation)" if include_generation_slope else "(1 | generation)"
    formula_parts = [
        "log_price ~ 1",
        "age",
        "mileage_scaled",
        generation_re,
        f"(1 | {trim_col})",
        f"(1 | {trans_col})",
    ]

    if include_sale_year:
        formula_parts.insert(3, "sale_year")

    if include_color and "color_category" in df.columns:
        color_idx = 4 if include_sale_year else 3
        formula_parts.insert(color_idx, "color_category")

    formula = " + ".join(formula_parts)
    logger.info(f"Model formula: {formula}")

    # Build priors
    prior_config = {**DEFAULT_RE_PRIORS, **(priors or {})}
    bambi_priors = _build_bambi_priors(prior_config, trim_col, trans_col, include_generation_slope)

    model = bmb.Model(formula, data=df, priors=bambi_priors, family="gaussian")
    logger.info(f"Built model with {len(df)} observations")

    return model


def _build_bambi_priors(
    config: dict[str, float], trim_col: str, trans_col: str, include_slope: bool
) -> dict[str, Any]:
    """Convert config dict to Bambi prior specifications.

    Args:
        config: Dict with SD values for HalfNormal priors
        trim_col: Name of trim column ('trim' or 'trim_tier')
        trans_col: Name of transmission column ('transmission' or 'trans_type')
        include_slope: Whether model includes age slope on generation

    Returns:
        Dict of Bambi Prior objects
    """
    priors = {
        "1|generation": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["generation_sd"])
        ),
        f"1|{trim_col}": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["trim_sd"])
        ),
        f"1|{trans_col}": bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["transmission_sd"])
        ),
    }

    if include_slope:
        priors["age|generation"] = bmb.Prior(
            "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=config["age_slope_sd"])
        )

    return priors


def fit_model(
    model: bmb.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 8,
    target_accept: float = 0.9,
    random_seed: int = 42,
    **kwargs,
) -> az.InferenceData:
    """Fit Bayesian model using MCMC sampling.

    Args:
        model: Bambi Model object
        draws: Number of posterior samples per chain
        tune: Number of tuning/warmup samples
        chains: Number of MCMC chains (for convergence diagnostics)
        target_accept: Target acceptance rate for NUTS sampler. Higher values
            (e.g., 0.95, 0.99) reduce divergences but slow sampling.
        random_seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to model.fit()

    Returns:
        ArviZ InferenceData with posterior samples and diagnostics
    """
    logger.info(
        f"Fitting model: {draws} draws, {tune} tune, {chains} chains, target_accept={target_accept}"
    )

    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        **kwargs,
    )

    # Log basic diagnostics
    summary = az.summary(idata, var_names=["~1|", "~age|"])  # Fixed effects only
    logger.info(f"Model summary:\n{summary}")

    # Check for convergence issues
    rhat_max = summary["r_hat"].max()
    if rhat_max > 1.01:
        logger.warning(f"Potential convergence issues: max R-hat = {rhat_max:.3f}")

    return idata


def check_diagnostics(idata: az.InferenceData) -> dict:
    """Check MCMC diagnostics for fitted model.

    Args:
        idata: InferenceData from fitted model

    Returns:
        Dict with diagnostic summaries
    """
    summary = az.summary(idata)

    diagnostics = {
        "n_divergences": idata.sample_stats.diverging.sum().item()
        if hasattr(idata, "sample_stats")
        else 0,
        "rhat_max": summary["r_hat"].max(),
        "rhat_above_105": (summary["r_hat"] > 1.01).sum(),
        "ess_bulk_min": summary["ess_bulk"].min(),
        "ess_tail_min": summary["ess_tail"].min(),
    }

    # Interpretation
    issues = []
    if diagnostics["n_divergences"] > 0:
        issues.append(f"{diagnostics['n_divergences']} divergences detected")
    if diagnostics["rhat_max"] > 1.01:
        issues.append(f"High R-hat: {diagnostics['rhat_max']:.3f}")
    if diagnostics["ess_bulk_min"] < 400:
        issues.append(f"Low ESS: {diagnostics['ess_bulk_min']:.0f}")

    diagnostics["issues"] = issues
    diagnostics["converged"] = len(issues) == 0

    return diagnostics


def predict_price(
    model: bmb.Model,
    idata: az.InferenceData,
    generation: str,
    trim: str,
    transmission: str,
    model_year: int,
    mileage: int,
    sale_year: int,
    mileage_mean: float,
    mileage_std: float,
    trim_tier: str | None = None,
    trans_type: str | None = None,
) -> dict:
    """Predict price distribution for a specific car configuration.

    Uses posterior predictive distribution to get full uncertainty.

    Args:
        model: Fitted Bambi model
        idata: InferenceData from model fitting
        generation: Car generation (e.g., "992.1")
        trim: Trim level (e.g., "Carrera 4S") - used for display
        transmission: Transmission type (e.g., "PDK") - used for display
        model_year: Model year of car
        mileage: Current mileage
        sale_year: Year of (hypothetical) sale
        mileage_mean: Mean mileage from training data (for scaling)
        mileage_std: Std dev of mileage from training data (for scaling)
        trim_tier: If model uses trim_tier, provide directly (e.g., "sport").
            If None, uses trim column (for models without grouping).
        trans_type: If model uses trans_type, provide directly (e.g., "manual").
            If None, uses transmission column (for models without grouping).

    Returns:
        Dict with price predictions and uncertainty intervals
    """
    age = sale_year - model_year
    mileage_scaled = (mileage - mileage_mean) / mileage_std

    new_data = pd.DataFrame(
        {
            "age": [age],
            "mileage_scaled": [mileage_scaled],
            "sale_year": [sale_year],
            "generation": pd.Categorical([generation]),
        }
    )

    # Handle trim vs trim_tier based on what model expects
    if trim_tier is not None:
        new_data["trim_tier"] = pd.Categorical([trim_tier])
    else:
        new_data["trim"] = pd.Categorical([trim])

    # Handle transmission vs trans_type based on what model expects
    if trans_type is not None:
        new_data["trans_type"] = pd.Categorical([trans_type])
    else:
        new_data["transmission"] = pd.Categorical([transmission])

    # Get posterior predictive samples
    model.predict(idata, data=new_data, kind="response", inplace=True)

    # Extract samples (shape: chains x draws)
    log_price_samples = idata.posterior_predictive["log_price"].values.flatten()
    price_samples = np.exp(log_price_samples)

    return {
        "config": {
            "generation": generation,
            "trim": trim,
            "transmission": transmission,
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


def extract_effects(idata: az.InferenceData) -> dict:
    """Extract interpretable effect estimates from fitted model.

    Returns posterior summaries for:
    - Fixed effects (age, mileage, sale_year)
    - Random effect standard deviations
    - Group-level effects (generation, trim, transmission)

    Args:
        idata: InferenceData from fitted model

    Returns:
        Dict with effect summaries
    """
    summary = az.summary(idata, hdi_prob=0.9)

    effects = {
        "fixed": {},
        "random_sd": {},
        "generation": {},
        "trim": {},
        "transmission": {},
    }

    for param in summary.index:
        mean = summary.loc[param, "mean"]
        hdi_low = summary.loc[param, "hdi_5%"]
        hdi_high = summary.loc[param, "hdi_95%"]

        result = {"mean": mean, "hdi_90": [hdi_low, hdi_high]}

        # Categorize parameters
        if param in ["Intercept", "age", "mileage_scaled", "sale_year"]:
            effects["fixed"][param] = result
        elif "sigma" in param.lower() or "_sd" in param:
            effects["random_sd"][param] = result
        elif "generation" in param:
            # Extract generation name from param
            gen = param.split("[")[-1].rstrip("]") if "[" in param else param
            effects["generation"][gen] = result
        elif "trim" in param:
            trim = param.split("[")[-1].rstrip("]") if "[" in param else param
            effects["trim"][trim] = result
        elif "transmission" in param:
            trans = param.split("[")[-1].rstrip("]") if "[" in param else param
            effects["transmission"][trans] = result

    return effects


def format_prediction_summary(pred: dict) -> str:
    """Format prediction result as human-readable summary.

    Args:
        pred: Output from predict_price()

    Returns:
        Formatted string summary
    """
    cfg = pred["config"]
    p = pred["price"]

    lines = [
        "Price prediction for:",
        f"  {cfg['model_year']} {cfg['generation']} {cfg['trim']} ({cfg['transmission']})",
        f"  Mileage: {cfg['mileage']:,} miles",
        f"  Sale year: {cfg['sale_year']} (age: {cfg['age']} years)",
        "",
        "Predicted price:",
        f"  Median: ${p['median']:,.0f}",
        f"  Mean:   ${p['mean']:,.0f}",
        f"  80% CI: ${p['ci_80'][0]:,.0f} - ${p['ci_80'][1]:,.0f}",
        f"  95% CI: ${p['ci_95'][0]:,.0f} - ${p['ci_95'][1]:,.0f}",
    ]

    return "\n".join(lines)
