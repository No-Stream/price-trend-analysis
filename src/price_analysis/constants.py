"""Central constants module for price analysis.

This module consolidates constants used across the project, including:
- Generation and trim classifications
- Mappings for EDA groupings
- Scraping configuration defaults
- Re-exports from cleaning.py for convenience
"""

# Re-export validation constants from cleaning.py
from price_analysis.data.cleaning import (
    TRANS_TO_TYPE,
    TRIM_TO_TIER,
    VALID_GENERATIONS,
    VALID_TRANSMISSIONS,
    VALID_TRIMS,
)

# Generation classifications
WATER_COOLED_GENS = ["996.1", "996.2", "997.1", "997.2", "991.1", "991.2", "992.1", "992.2"]

GT_TRIMS = ["GT3", "GT3 RS", "GT3 Touring", "GT2 RS"]

# EDA trim mapping for grouping sparse trims into more common variants
# Targas -> AWD coupe equivalents, Turbo S -> Turbo
TRIM_MAPPING = {
    "Targa": "Carrera 4",
    "Targa 4": "Carrera 4",
    "Targa 4S": "Carrera 4S",
    "Targa 4 GTS": "Carrera 4 GTS",
    "Turbo S": "Turbo",
}

# Scraping configuration defaults
SCRAPING_DEFAULTS = {
    "max_clicks": 50,
    "delay": 1.0,
    "headless": True,
}

__all__ = [
    # Re-exports from cleaning.py
    "VALID_GENERATIONS",
    "VALID_TRIMS",
    "VALID_TRANSMISSIONS",
    "TRIM_TO_TIER",
    "TRANS_TO_TYPE",
    # Generation/trim classifications
    "WATER_COOLED_GENS",
    "GT_TRIMS",
    # EDA mappings
    "TRIM_MAPPING",
    # Scraping config
    "SCRAPING_DEFAULTS",
]
