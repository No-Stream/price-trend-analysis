"""Smoke tests for Bayesian hierarchical model."""

import pandas as pd
import pytest

from price_analysis.models.hierarchical import (
    build_model,
    check_diagnostics,
    extract_effects,
    fit_model,
    format_prediction_summary,
    predict_price,
)


class TestModelSmoke:
    """Smoke tests - model builds and runs without crashing.

    These tests use minimal iterations to run fast. They verify
    the code doesn't crash, not that the model produces good results.
    """

    def test_model_builds(self, minimal_model_data: pd.DataFrame):
        """Model builds without error."""
        model = build_model(minimal_model_data)
        assert model is not None
        assert model.formula is not None

    @pytest.mark.slow
    def test_model_fits(self, minimal_model_data: pd.DataFrame):
        """Model fits with minimal iterations (smoke test).

        This test is marked slow since even minimal MCMC takes a few seconds.
        Run with: pytest -m slow
        """
        model = build_model(minimal_model_data)
        idata = fit_model(model, draws=10, tune=10, chains=1)

        assert idata is not None
        assert "posterior" in idata.groups()

    @pytest.mark.slow
    def test_diagnostics_run(self, minimal_model_data: pd.DataFrame):
        """Diagnostic functions run without error."""
        model = build_model(minimal_model_data)
        idata = fit_model(model, draws=10, tune=10, chains=1)

        diagnostics = check_diagnostics(idata)

        assert "converged" in diagnostics
        assert "n_divergences" in diagnostics
        assert "rhat_max" in diagnostics

    @pytest.mark.slow
    def test_prediction_runs(self, minimal_model_data: pd.DataFrame):
        """Prediction function runs without error."""
        model = build_model(minimal_model_data)
        idata = fit_model(model, draws=10, tune=10, chains=1)

        result = predict_price(
            model=model,
            idata=idata,
            generation="992.1",
            trim="Carrera 4S",
            transmission="PDK",
            model_year=2022,
            mileage=15000,
            sale_year=2025,
            mileage_mean=30000,
            mileage_std=20000,
        )

        assert "price" in result
        assert "config" in result
        assert result["price"]["median"] > 0
        assert len(result["price"]["ci_80"]) == 2

    @pytest.mark.slow
    def test_extract_effects_runs(self, minimal_model_data: pd.DataFrame):
        """Effect extraction runs without error."""
        model = build_model(minimal_model_data)
        idata = fit_model(model, draws=10, tune=10, chains=1)

        effects = extract_effects(idata)

        assert "fixed" in effects
        assert "generation" in effects
        assert "trim" in effects
        assert "transmission" in effects

    @pytest.mark.slow
    def test_format_prediction_summary(self, minimal_model_data: pd.DataFrame):
        """Prediction summary formatting works."""
        model = build_model(minimal_model_data)
        idata = fit_model(model, draws=10, tune=10, chains=1)

        pred = predict_price(
            model=model,
            idata=idata,
            generation="992.1",
            trim="Carrera 4S",
            transmission="PDK",
            model_year=2022,
            mileage=15000,
            sale_year=2025,
            mileage_mean=30000,
            mileage_std=20000,
        )

        summary = format_prediction_summary(pred)

        assert isinstance(summary, str)
        assert "992.1" in summary
        assert "Carrera 4S" in summary
        assert "$" in summary


class TestModelBuilding:
    """Test model building with different configurations."""

    def test_model_with_color(self, minimal_model_data: pd.DataFrame):
        """Model builds with color_category included."""
        # Add color category
        df = minimal_model_data.copy()
        df["color_category"] = pd.Categorical(["standard", "special", "PTS"] * 10)

        model = build_model(df, include_color=True)
        assert model is not None

    def test_model_requires_all_columns(self):
        """Model building fails gracefully with missing columns."""
        incomplete_df = pd.DataFrame({"log_price": [11.5, 11.6], "age": [2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            build_model(incomplete_df)
