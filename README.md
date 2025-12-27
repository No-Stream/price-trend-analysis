# Price Trend Analysis

Bayesian hierarchical modeling of Porsche 911 auction prices from Bring a Trailer.
Includes scraper, python modules, and analysis notebooks for a few models (e.g. hierarchical/mixed effect linear and spline.)
Data included for your use! Mostly 2025 911 auctions but some 2024.

## Quick Start

```bash
# Create environment (requires mamba)
~/miniconda3/bin/mamba env create -f environment.yml
conda activate price-analysis

# Install package
~/.local/bin/uv pip install -e . --python /opt/homebrew/Caskroom/miniconda/base/envs/price-analysis/bin/python

# Run tests
make test
```

## Project Structure

```
src/price_analysis/
├── scraping/bat.py        # BaT Selenium scraper
├── data/cleaning.py       # Validation, feature engineering
├── models/
│   ├── hierarchical.py    # Bayesian hierarchical model (Bambi/PyMC)
│   ├── spline.py          # B-spline model variant
│   └── comparison.py      # LOO-CV, residual diagnostics
├── visualization/
│   ├── eda_viz.py         # EDA plots
│   └── model_viz.py       # Model diagnostics
└── constants.py           # Generation/trim mappings

notebooks/
├── 01_scraping.ipynb      # Run scraper, inspect raw data
├── 02_eda.ipynb           # Exploration, validation, cleaning
├── 03_modeling.ipynb      # Fit hierarchical model, diagnostics
└── 04_splines.ipynb       # Compare model parameterizations
```

## Model

```
log(price) ~ age + mileage + sale_year
             + (1 + age | generation)   # Random intercept + slope
             + (1 | trim_tier)          # Random intercept
             + (1 | trans_type)         # Random intercept
             + (1 | body_style)
```

Crossed (not nested) random effects share information across groups. Generation gets varying slopes because depreciation rates differ (992.1 vs air-cooled). Trim/transmission premiums are more stable.

## Commands

| Command | Description |
|---------|-------------|
| `make test` | Run all tests (~30s) |
| `make test-fast` | Skip model fitting tests (~3s) |
| `make format` | Auto-format with ruff |
| `make lint` | Check formatting/lint |
| `make check` | lint + test |
| `make env` | Create conda environment |

## Data Fields

| Field | Description |
|-------|-------------|
| `listing_url` | BaT listing URL |
| `sale_price` | Final sale price (USD) |
| `sale_date` | Auction close date |
| `model_year` | Vehicle model year |
| `generation` | 996.1, 996.2, 997.1, 997.2, 991.1, 991.2, 992.1, 992.2 |
| `trim` | Carrera, Carrera S, Carrera 4S, GT3, Turbo, etc. |
| `transmission` | Manual, PDK, Tiptronic, Automatic |
| `mileage` | Odometer reading |
| `color` | Exterior color |

**Derived:** `age`, `log_price`, `mileage_scaled`, `trim_tier`, `trans_type`, `color_category`
