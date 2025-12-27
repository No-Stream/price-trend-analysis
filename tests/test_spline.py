"""Smoke tests for spline-based Bayesian model."""

import pandas as pd
import pytest

from price_analysis.models.comparison import (
    compute_residual_stats,
    get_residuals,
)
from price_analysis.models.hierarchical import check_diagnostics
from price_analysis.models.spline import (
    build_spline_model,
    fit_spline_model,
    plot_spline_effect,
    plot_spline_effects_grid,
)


class TestSplineModelSmoke:
    """Smoke tests - model builds and runs without crashing."""

    def test_spline_model_builds(self, minimal_spline_model_data: pd.DataFrame):
        """Spline model builds without error."""
        model = build_spline_model(minimal_spline_model_data)
        assert model is not None
        assert model.formula is not None
        assert "bs(age" in str(model.formula)
        assert "bs(mileage_scaled" in str(model.formula)

    def test_spline_model_builds_custom_df(self, minimal_spline_model_data: pd.DataFrame):
        """Spline model builds with custom degrees of freedom."""
        model = build_spline_model(minimal_spline_model_data, age_df=3, mileage_df=5)
        assert model is not None
        assert "df=3" in str(model.formula)
        assert "df=5" in str(model.formula)

    @pytest.mark.slow
    def test_spline_model_fits(self, minimal_spline_model_data: pd.DataFrame):
        """Spline model fits with minimal iterations (smoke test)."""
        model = build_spline_model(minimal_spline_model_data)
        idata = fit_spline_model(model, draws=10, tune=10, chains=1)

        assert idata is not None
        assert "posterior" in idata.groups()

    @pytest.mark.slow
    def test_spline_diagnostics_run(self, minimal_spline_model_data: pd.DataFrame):
        """Diagnostic functions run on spline model."""
        model = build_spline_model(minimal_spline_model_data)
        idata = fit_spline_model(model, draws=10, tune=10, chains=1)

        diagnostics = check_diagnostics(idata)

        assert "converged" in diagnostics
        assert "n_divergences" in diagnostics


class TestSplineModelBuilding:
    """Test spline model building with different configurations."""

    def test_model_requires_trim_tier(self, minimal_model_data: pd.DataFrame):
        """Model building fails without trim_tier column."""
        with pytest.raises(ValueError, match="Missing required columns"):
            build_spline_model(minimal_model_data)

    def test_model_without_sale_year(self, minimal_spline_model_data: pd.DataFrame):
        """Model builds without sale_year when disabled."""
        model = build_spline_model(minimal_spline_model_data, include_sale_year=False)
        assert model is not None
        assert "sale_year" not in str(model.formula)


class TestSplineVisualization:
    """Test spline visualization functions."""

    @pytest.mark.slow
    def test_plot_spline_effect_runs(self, minimal_spline_model_data: pd.DataFrame):
        """Spline effect plot runs without error."""
        import matplotlib.pyplot as plt

        model = build_spline_model(minimal_spline_model_data)
        idata = fit_spline_model(model, draws=10, tune=10, chains=1)

        ax = plot_spline_effect(model, idata, minimal_spline_model_data, "age")

        assert ax is not None
        plt.close("all")

    @pytest.mark.slow
    def test_plot_spline_effects_grid_runs(self, minimal_spline_model_data: pd.DataFrame):
        """Spline effects grid plot runs without error."""
        import matplotlib.pyplot as plt

        model = build_spline_model(minimal_spline_model_data)
        idata = fit_spline_model(model, draws=10, tune=10, chains=1)

        fig = plot_spline_effects_grid(model, idata, minimal_spline_model_data)

        assert fig is not None
        plt.close("all")


class TestComparisonUtilities:
    """Test model comparison utilities."""

    @pytest.mark.slow
    def test_get_residuals_runs(self, minimal_spline_model_data: pd.DataFrame):
        """Residuals extraction runs without error."""
        model = build_spline_model(minimal_spline_model_data)
        idata = fit_spline_model(model, draws=10, tune=10, chains=1)

        residuals = get_residuals(model, idata, minimal_spline_model_data)

        assert "residual_mean" in residuals.columns
        assert "observed" in residuals.columns
        assert "predicted_mean" in residuals.columns
        assert len(residuals) == len(minimal_spline_model_data)

    @pytest.mark.slow
    def test_compute_residual_stats_runs(self, minimal_spline_model_data: pd.DataFrame):
        """Residual stats computation runs without error."""
        model = build_spline_model(minimal_spline_model_data)
        idata = fit_spline_model(model, draws=10, tune=10, chains=1)

        residuals = get_residuals(model, idata, minimal_spline_model_data)
        stats = compute_residual_stats(residuals)

        assert "rmse" in stats
        assert "mae" in stats
        assert stats["rmse"] >= 0
