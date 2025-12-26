"""Tests for data cleaning pipeline."""

import pandas as pd
import pytest

from price_analysis.data.cleaning import (
    categorize_color,
    clean_listings,
    get_summary_stats,
    prepare_model_data,
    validate_listing,
)


class TestCleaningPipeline:
    """Test full cleaning pipeline with realistic data."""

    def test_clean_listings_adds_derived_features(self, sample_listings_df: pd.DataFrame):
        """Cleaning adds age, mileage_scaled, log_price, etc."""
        result = clean_listings(sample_listings_df)

        assert "age" in result.columns
        assert "mileage_scaled" in result.columns
        assert "log_price" in result.columns
        assert "color_category" in result.columns
        assert "sale_year" in result.columns
        assert "mileage_10k" in result.columns

    def test_clean_listings_validates_records(self, sample_listings_df: pd.DataFrame):
        """Validation flags records and adds is_valid column."""
        result = clean_listings(sample_listings_df)

        assert "is_valid" in result.columns
        # Should have some valid records
        assert result["is_valid"].sum() > 0
        # Sample data has some incomplete records
        assert result["is_valid"].sum() < len(result)

    def test_clean_listings_deduplicates(self, sample_listings_df: pd.DataFrame):
        """Duplicate listing_urls are removed."""
        # Add a duplicate
        df_with_dup = pd.concat([sample_listings_df, sample_listings_df.iloc[[0]]])
        assert len(df_with_dup) == len(sample_listings_df) + 1

        result = clean_listings(df_with_dup)
        assert len(result) == len(sample_listings_df)

    def test_clean_listings_drop_invalid(self, sample_listings_df: pd.DataFrame):
        """drop_invalid=True removes invalid records."""
        result_keep = clean_listings(sample_listings_df, drop_invalid=False)
        result_drop = clean_listings(sample_listings_df, drop_invalid=True)

        # Should have fewer records when dropping
        assert len(result_drop) <= len(result_keep)
        # All remaining should be valid
        assert result_drop["is_valid"].all()

    def test_age_calculation(self, sample_listings_df: pd.DataFrame):
        """Age correctly calculated as sale_year - model_year."""
        result = clean_listings(sample_listings_df)

        # Check a specific row
        row = result[result["model_year"] == 2022].iloc[0]
        expected_age = row["sale_year"] - 2022
        assert row["age"] == expected_age


class TestValidateListing:
    """Test individual listing validation."""

    def test_valid_listing_passes(self, sample_listings_df: pd.DataFrame):
        """Complete valid listing passes validation."""
        row = sample_listings_df.iloc[0]  # First row is complete
        result = validate_listing(row)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_price_fails(self):
        """Missing sale_price causes validation error."""
        row = pd.Series(
            {
                "sale_price": None,
                "model_year": 2022,
                "mileage": 15000,
                "generation": "992.1",
                "trim": "Carrera 4S",
                "transmission": "PDK",
            }
        )
        result = validate_listing(row)

        assert not result.is_valid
        assert any("sale_price" in e for e in result.errors)

    def test_negative_mileage_fails(self):
        """Negative mileage causes validation error."""
        row = pd.Series(
            {
                "sale_price": 100000,
                "model_year": 2022,
                "mileage": -5000,
                "generation": "992.1",
                "trim": "Carrera 4S",
                "transmission": "PDK",
            }
        )
        result = validate_listing(row)

        assert not result.is_valid
        assert any("mileage" in e.lower() for e in result.errors)

    def test_unusual_price_warns(self):
        """Unusually low or high prices generate warnings."""
        row = pd.Series(
            {
                "sale_price": 14999,  # Very low for a 911 (below $15k threshold)
                "model_year": 2022,
                "mileage": 15000,
                "generation": "992.1",
                "trim": "Carrera 4S",
                "transmission": "PDK",
            }
        )
        result = validate_listing(row)

        # Should still be valid, just warned
        assert result.is_valid
        assert len(result.warnings) > 0


class TestCategorizeColor:
    """Test color categorization."""

    @pytest.mark.parametrize(
        "color,expected",
        [
            ("PTS Rubystone Red", "PTS"),
            ("Paint to Sample Signal Green", "PTS"),
            ("Miami Blue", "special"),
            ("GT Silver Metallic", "special"),
            ("Guards Red", "special"),
            ("Black", "standard"),
            ("White", "standard"),
            ("Carrara White", "standard"),
            (None, "unknown"),
        ],
    )
    def test_color_categorization(self, color: str | None, expected: str):
        """Colors correctly categorized as PTS, special, or standard."""
        assert categorize_color(color) == expected


class TestPrepareModelData:
    """Test model data preparation."""

    def test_prepare_model_data_drops_incomplete(self, sample_listings_df: pd.DataFrame):
        """Model prep drops rows missing required fields."""
        cleaned = clean_listings(sample_listings_df)
        model_df = prepare_model_data(cleaned)

        required = [
            "log_price",
            "age",
            "mileage_scaled",
            "generation",
            "trim",
            "transmission",
        ]
        # All required cols should be non-null
        assert model_df[required].notna().all().all()

    def test_prepare_model_data_converts_categoricals(self, sample_listings_df: pd.DataFrame):
        """Categorical columns converted to proper dtype."""
        cleaned = clean_listings(sample_listings_df)
        model_df = prepare_model_data(cleaned)

        for col in ["generation", "trim", "transmission"]:
            assert model_df[col].dtype.name == "category"


class TestGetSummaryStats:
    """Test summary statistics function."""

    def test_summary_stats_structure(self, sample_listings_df: pd.DataFrame):
        """Summary stats returns expected structure."""
        cleaned = clean_listings(sample_listings_df)
        stats = get_summary_stats(cleaned)

        assert "n_listings" in stats
        assert "price_median" in stats
        assert "by_generation" in stats
        assert "by_trim" in stats
        assert "by_transmission" in stats
        assert isinstance(stats["by_generation"], dict)
