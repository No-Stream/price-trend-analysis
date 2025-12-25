.PHONY: format lint test check clean install

# Format code with ruff
format:
	ruff format src tests notebooks
	ruff check --fix src tests

# Lint code with ruff
lint:
	ruff check src tests
	ruff format --check src tests notebooks

# Run tests
test:
	pytest tests -v

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
