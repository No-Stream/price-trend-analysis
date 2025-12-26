# Price Trend Analysis - Project Guidelines

## Project Overview

Bayesian hierarchical modeling of Porsche 911 auction prices from Bring a Trailer. Focus on 992.1 Carrera 4S but tracking multiple generations/trims.

**Stack:** Python 3.12, PyMC, Bambi, ArviZ, Selenium, Pandas

## Environment

```bash
# Create environment (use mamba, not base conda)
~/miniconda3/bin/mamba env create -f environment.yml

# Activate (use conda activate, not mamba activate)
conda activate price-analysis

# Add new packages (use uv, not pip)
~/.local/bin/uv pip install <package> --python /opt/homebrew/Caskroom/miniconda/base/envs/price-analysis/bin/python

# Install this package in editable mode
~/.local/bin/uv pip install -e . --python /opt/homebrew/Caskroom/miniconda/base/envs/price-analysis/bin/python
```

**Package management rules:**
- Environment creation/updates: `mamba` (fast solver)
- Adding packages: `uv pip install` (fast, respects env)
- **NEVER** use base `pip` or `conda` resolvers

## Project Structure

```
src/price_analysis/
├── scraping/bat.py      # BaT scraper (Selenium)
├── data/cleaning.py     # Validation, feature engineering
└── models/hierarchical.py  # Bambi model with documented intuitions

notebooks/
├── 01_scraping.ipynb    # Run scraper, inspect results
├── 02_eda.ipynb         # Exploration, validation
└── 03_modeling.ipynb    # Fit, diagnose, visualize
```

## Commands

```bash
# Testing
make test        # Run all tests including slow model fits (~30s)
make test-fast   # Skip slow tests (~3s) - use for quick iteration

# Code quality
make format      # Auto-format with ruff (also fixes lint issues)
make lint        # Check formatting and lint (no changes)
make check       # lint + test (CI pipeline)

# Environment
make env         # Create conda environment from environment.yml
make env-update  # Update environment (use after editing environment.yml)
make install     # Install package in editable mode
make clean       # Remove __pycache__, .pyc, etc.
```

**Workflow:** `make format` before commits, `make test-fast` during dev, `make check` before PR.

## Key Design Decisions

### Bayesian Model Structure

```python
log_price ~ age + mileage_scaled + sale_year +
            (1 + age | generation) +  # Random intercept + slope
            (1 | trim_tier) +         # Random intercept only (or trim if enough data)
            (1 | trans_type)          # Random intercept only (or transmission)
```

**Sparse data options** (use when sample sizes per group < 20):
- `use_trim_tier=True`: Collapse trims → 4 tiers (base, sport, gt, turbo)
- `use_trans_type=True`: Collapse transmissions → 3 types (manual, pdk, auto)
- Weakly informative priors on RE SDs (HalfNormal σ=0.3-0.5)

**Why crossed (not nested) random effects:**
- "Manual premium" and "4S premium" are relatively consistent across generations
- Crossed effects share information across groups (more parsimonious)
- Can add interaction terms later if residuals suggest

**Why random slopes on generation only:**
- Different generations depreciate at different rates (992.1 vs air-cooled)
- Trim/transmission premiums are more stable over time

### Data Fields

Required: `listing_url`, `sale_price`, `sale_date`, `model_year`, `generation`, `trim`, `transmission`, `mileage`

Derived:
- `age` (sale_year - model_year)
- `mileage_scaled` (z-scored)
- `log_price`
- `color_category` (PTS, special, standard)
- `trim_tier` (base, sport, gt, turbo) - optional grouping
- `trans_type` (manual, pdk, auto) - optional grouping

### Generation Mapping

| Years | Generation |
|-------|------------|
| 1999-2001 | 996.1 |
| 2002-2004 | 996.2 |
| 2005-2008 | 997.1 |
| 2009-2012 | 997.2 |
| 2012-2015 | 991.1 |
| 2016-2019 | 991.2 |
| 2020-2023 | 992.1 |
| 2024+ | 992.2 |

## Testing Philosophy

- **Integration over unit tests** - test pipelines end-to-end
- **Realistic fixtures** - sample data covering generations, trims, edge cases
- **Model smoke tests** - verify builds/fits without crashing (minimal iterations)
- Run `make test` by default (includes slow tests)

## Future Enhancements (Not MVP)

1. Student-t likelihood (robustness)
2. Splines on age (non-linear depreciation)
3. Gaussian Process for age
4. Options parsing from descriptions
5. Interaction terms if residuals suggest

## Scraping Notes

- Based on `KaledDahleh/bring-a-trailer-tracker` approach
- Uses Selenium (BaT requires JS rendering)
- Rate limit: 1.0s between requests (configurable via `delay` param)
- Progress bar via tqdm shows ETA
- Selectors may need adjustment based on BaT DOM changes
