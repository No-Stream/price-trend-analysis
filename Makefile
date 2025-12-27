.PHONY: format lint test test-fast test-slow check clean install env env-update

# Conda environment path
CONDA_ENV := /opt/homebrew/Caskroom/miniconda/base/envs/price-analysis/bin

# Format code with ruff
format:
	$(CONDA_ENV)/ruff format src tests notebooks
	$(CONDA_ENV)/ruff check --fix src tests

# Lint code with ruff
lint:
	$(CONDA_ENV)/ruff check src tests
	$(CONDA_ENV)/ruff format --check src tests notebooks

# Run all tests (default, ~30s with model fits)
test:
	$(CONDA_ENV)/pytest tests -v

# Run fast tests only (skip model fitting, ~3s)
test-fast:
	$(CONDA_ENV)/pytest tests -v -m "not slow"

# Run slow tests only (model fitting tests)
test-slow:
	$(CONDA_ENV)/pytest tests -v -m slow

# Run all checks (format + lint + test)
check: lint test

# Install package in editable mode
install:
	pip install -e ".[dev]"

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Create conda environment
env:
	conda env create -f environment.yml

# Update conda environment
env-update:
	conda env update -f environment.yml --prune
