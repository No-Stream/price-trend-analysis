"""Data cleaning and validation for BaT auction listings."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

# Valid values for categorical fields
VALID_GENERATIONS = {
    "996.1",
    "996.2",
    "997.1",
    "997.2",
    "991.1",
    "991.2",
    "992.1",
    "992.2",
}

VALID_TRIMS = {
    "Carrera",
    "Carrera S",
    "Carrera 4",
    "Carrera 4S",
    "Targa",
    "Targa 4",
    "Targa 4S",
    "Turbo",
    "Turbo S",
    "GT3",
    "GT3 RS",
    "GT3 Touring",
    "GT2 RS",
}

VALID_TRANSMISSIONS = {"PDK", "Manual", "Tiptronic", "Automatic"}

# Special colors that command premiums
PTS_KEYWORDS = ["pts", "paint to sample", "special order"]
SPECIAL_COLORS = {
    "gt silver",
    "miami blue",
    "shark blue",
    "python green",
    "riviera blue",
    "signal green",
    "guards red",
    "speed yellow",
    "racing yellow",
    "chalk",
    "crayon",
    "graphite blue",
    "gentian blue",
}


@dataclass
class ValidationResult:
    """Result of validating a single listing."""

    is_valid: bool
    warnings: list[str]
    errors: list[str]


def validate_listing(row: pd.Series) -> ValidationResult:
    """Validate a single listing row.

    Args:
        row: DataFrame row representing a listing

    Returns:
        ValidationResult with validity flag and any warnings/errors
    """
    warnings: list[str] = []
    errors: list[str] = []

    # Required fields
    if pd.isna(row.get("sale_price")) or row["sale_price"] <= 0:
        errors.append("Invalid or missing sale_price")

    if pd.isna(row.get("model_year")):
        errors.append("Missing model_year")
    elif row["model_year"] < 1999 or row["model_year"] > 2030:
        warnings.append(f"Unusual model_year: {row['model_year']}")

    if pd.isna(row.get("mileage")):
        warnings.append("Missing mileage")
    elif row["mileage"] < 0:
        errors.append(f"Negative mileage: {row['mileage']}")
    elif row["mileage"] > 300000:
        warnings.append(f"Very high mileage: {row['mileage']}")

    # Categorical validation
    if pd.notna(row.get("generation")) and row["generation"] not in VALID_GENERATIONS:
        warnings.append(f"Unknown generation: {row['generation']}")

    if pd.notna(row.get("trim")) and row["trim"] not in VALID_TRIMS:
        warnings.append(f"Unknown trim: {row['trim']}")

    if pd.notna(row.get("transmission")) and row["transmission"] not in VALID_TRANSMISSIONS:
        warnings.append(f"Unknown transmission: {row['transmission']}")

    # Price sanity checks (911-specific)
    if pd.notna(row.get("sale_price")):
        price = row["sale_price"]
        if price < 20000:
            warnings.append(f"Unusually low price: ${price:,}")
        if price > 500000 and row.get("trim") not in {"GT2 RS", "GT3 RS"}:
            warnings.append(f"Unusually high price: ${price:,}")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, warnings=warnings, errors=errors)


def categorize_color(color: str | None) -> str:
    """Categorize color into PTS, special, or standard.

    Args:
        color: Raw color string from listing

    Returns:
        Category: "PTS", "special", or "standard"
    """
    if color is None or pd.isna(color):
        return "unknown"

    color_lower = color.lower()

    # Check for PTS indicators
    if any(kw in color_lower for kw in PTS_KEYWORDS):
        return "PTS"

    # Check for known special colors
    if any(sc in color_lower for sc in SPECIAL_COLORS):
        return "special"

    return "standard"


def clean_listings(df: pd.DataFrame, drop_invalid: bool = False) -> pd.DataFrame:
    """Clean and validate listings DataFrame.

    Performs:
    - Type conversions
    - Feature engineering (age, mileage_scaled, color_category)
    - Validation (warnings logged, optionally drop invalid)
    - Deduplication by listing_url

    Args:
        df: Raw listings DataFrame
        drop_invalid: If True, drop rows that fail validation

    Returns:
        Cleaned DataFrame with additional derived columns
    """
    df = df.copy()
    logger.info(f"Cleaning {len(df)} listings")

    # Deduplicate
    initial_count = len(df)
    df = df.drop_duplicates(subset=["listing_url"], keep="last")
    if len(df) < initial_count:
        logger.info(f"Removed {initial_count - len(df)} duplicate listings")

    # Type conversions
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce").astype("Int64")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce").astype("Int64")

    # Derived features
    df["sale_year"] = df["sale_date"].dt.year
    df["age"] = df["sale_year"] - df["model_year"]

    # Mileage scaling (per 10k miles for interpretability)
    df["mileage_10k"] = df["mileage"] / 10000

    # Z-score scaling (computed on non-null values)
    mileage_mean = df["mileage"].mean()
    mileage_std = df["mileage"].std()
    df["mileage_scaled"] = (df["mileage"] - mileage_mean) / mileage_std

    # Color categorization
    df["color_category"] = df["color"].apply(categorize_color)

    # Log-transformed price (for modeling)
    df["log_price"] = np.log(df["sale_price"].replace(0, np.nan))

    # Validation
    valid_mask = []
    for idx, row in df.iterrows():
        result = validate_listing(row)
        valid_mask.append(result.is_valid)

        if result.warnings:
            logger.warning(f"Listing {row.get('listing_url', idx)}: {result.warnings}")
        if result.errors:
            logger.error(f"Listing {row.get('listing_url', idx)}: {result.errors}")

    df["is_valid"] = valid_mask
    invalid_count = (~df["is_valid"]).sum()

    if invalid_count > 0:
        logger.warning(f"{invalid_count} listings failed validation")

    if drop_invalid:
        df = df[df["is_valid"]].copy()
        logger.info(f"Dropped {invalid_count} invalid listings, {len(df)} remaining")

    logger.info(f"Cleaning complete. Final dataset: {len(df)} listings")
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Get summary statistics for cleaned dataset.

    Args:
        df: Cleaned listings DataFrame

    Returns:
        Dict with summary statistics
    """
    return {
        "n_listings": len(df),
        "n_valid": df["is_valid"].sum() if "is_valid" in df.columns else len(df),
        "date_range": (df["sale_date"].min(), df["sale_date"].max()),
        "price_range": (df["sale_price"].min(), df["sale_price"].max()),
        "price_median": df["sale_price"].median(),
        "mileage_median": df["mileage"].median(),
        "by_generation": df["generation"].value_counts().to_dict(),
        "by_trim": df["trim"].value_counts().to_dict(),
        "by_transmission": df["transmission"].value_counts().to_dict(),
        "by_color_category": df["color_category"].value_counts().to_dict(),
    }


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare cleaned data for Bayesian modeling.

    Filters to valid records with all required fields for modeling.

    Args:
        df: Cleaned listings DataFrame

    Returns:
        DataFrame ready for Bambi model fitting
    """
    required_cols = [
        "log_price",
        "age",
        "mileage_scaled",
        "sale_year",
        "generation",
        "trim",
        "transmission",
    ]

    model_df = df.dropna(subset=required_cols).copy()

    # Convert categoricals to proper type for Bambi
    for col in ["generation", "trim", "transmission", "color_category"]:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype("category")

    logger.info(f"Model-ready dataset: {len(model_df)} listings (from {len(df)} cleaned)")

    return model_df
