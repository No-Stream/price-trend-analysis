# Price Trend Analysis - Project Guidelines

## Project Overview

Bayesian hierarchical modeling of Porsche 911 auction prices from Bring a Trailer. Focus on 992.1 Carrera 4S but tracking multiple generations/trims.

**Stack:** Python 3.12, PyMC, Bambi, ArviZ, Selenium, Pandas

## Environment

```bash
# Use mamba (not base conda) for environment management
~/miniconda3/bin/mamba env create -f environment.yml
~/miniconda3/bin/mamba activate price-analysis

# Use uv (not pip) for package installs
~/.local/bin/uv pip install -e . --python /opt/homebrew/Caskroom/miniconda/base/envs/price-analysis/bin/python
```

**NEVER use base pip or conda resolvers** - always use mamba or uv.

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
make test        # Run all tests (~30s)
make test-fast   # Skip model fitting tests (~3s)
make format      # Format with ruff
make lint        # Lint with ruff
make check       # lint + test
```

## Key Design Decisions

### Bayesian Model Structure

```python
log_price ~ age + mileage_scaled + sale_year +
            (1 + age | generation) +  # Random intercept + slope
            (1 | trim) +              # Random intercept only
            (1 | transmission)        # Random intercept only
```

**Why crossed (not nested) random effects:**
- "Manual premium" and "4S premium" are relatively consistent across generations
- Crossed effects share information across groups (more parsimonious)
- Can add interaction terms later if residuals suggest

**Why random slopes on generation only:**
- Different generations depreciate at different rates (992.1 vs air-cooled)
- Trim/transmission premiums are more stable over time

### Data Fields

Required: `listing_url`, `sale_price`, `sale_date`, `model_year`, `generation`, `trim`, `transmission`, `mileage`

Derived: `age` (sale_year - model_year), `mileage_scaled`, `log_price`, `color_category`

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
- Rate limit: 2-3 sec between requests
- Selectors may need adjustment based on BaT DOM changes
