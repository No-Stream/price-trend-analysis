"""Tests for BaT title and field parsing functions."""

import pytest

from price_analysis.scraping.bat import (
    is_non_car_listing,
    parse_generation,
    parse_mileage,
    parse_price,
    parse_transmission,
    parse_trim,
    parse_year,
)


class TestParseGeneration:
    """Test generation parsing from title + year."""

    @pytest.mark.parametrize(
        "title,year,expected",
        [
            ("2022 Porsche 911 Carrera 4S", 2022, "992.1"),
            ("2021 Porsche 911 Carrera S PDK", 2021, "992.1"),
            ("2020 Porsche 911 Carrera", 2020, "992.1"),
            ("2019 Porsche 911 GT3 RS", 2019, "991.2"),
            ("2017 Porsche 911 Turbo S Cabriolet", 2017, "991.2"),
            ("2014 Porsche 911 Carrera 4S", 2014, "991.1"),
            ("2010 Porsche 911 Carrera S", 2010, "997.2"),
            ("2007 Porsche 911 Turbo", 2007, "997.1"),
            ("2003 Porsche 911 Carrera 4S", 2003, "996.2"),
            ("2000 Porsche 911 Carrera", 2000, "996.1"),
        ],
    )
    def test_generation_from_year(self, title: str, year: int, expected: str):
        """Generation correctly inferred from model year."""
        assert parse_generation(title, year) == expected

    def test_explicit_generation_in_title_takes_precedence(self):
        """Explicit generation mention in title overrides year-based lookup."""
        # If title explicitly says 991.2, use that even if year is ambiguous
        result = parse_generation("2012 Porsche 911 991.1 Carrera", 2012)
        assert result == "991.1"


class TestParseTrim:
    """Test trim extraction from titles."""

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("2022 Porsche 911 Carrera 4S Cabriolet", "Carrera 4S"),
            ("2019 Porsche 911 GT3 RS Weissach Package", "GT3 RS"),
            ("2021 Porsche 911 Turbo S", "Turbo S"),
            ("2020 Porsche 911 Carrera", "Carrera"),
            ("2018 Porsche 911 Carrera S", "Carrera S"),
            ("2017 Porsche 911 Carrera 4", "Carrera 4"),
            ("2019 Porsche 911 GT3 Touring", "GT3 Touring"),
            ("2018 Porsche 911 GT3", "GT3"),
            ("2020 Porsche 911 Turbo Cabriolet", "Turbo"),
            ("2015 Porsche 911 Targa 4S", "Targa 4S"),
            ("2016 Porsche 911 Targa 4", "Targa 4"),
            ("2014 Porsche 911 Targa", "Targa"),
        ],
    )
    def test_trim_extraction(self, title: str, expected: str):
        """Trim correctly extracted from title."""
        assert parse_trim(title) == expected

    def test_more_specific_trim_wins(self):
        """More specific trim (e.g., Carrera 4S) matches before less specific (Carrera)."""
        # Carrera 4S should match, not just Carrera
        assert parse_trim("2022 Porsche 911 Carrera 4S") == "Carrera 4S"
        # GT3 RS should match, not just GT3
        assert parse_trim("2019 Porsche 911 GT3 RS") == "GT3 RS"

    def test_shorthand_trim_codes(self):
        """Common shorthand codes (C4S, C2S) are parsed."""
        assert parse_trim("2010 Porsche 911 C4S") == "Carrera 4S"
        assert parse_trim("2010 Porsche 911 C2S") == "Carrera S"


class TestParseTransmission:
    """Test transmission extraction."""

    @pytest.mark.parametrize(
        "title,details,expected",
        [
            ("2022 Porsche 911 Carrera 4S PDK", "", "PDK"),
            ("2020 Porsche 911 Carrera S", "6-Speed Manual", "Manual"),
            ("2019 Porsche 911 GT3", "7-Speed Manual Transmission", "Manual"),
            ("2021 Porsche 911 Turbo S PDK", "Dual-clutch", "PDK"),
            ("2015 Porsche 911 Carrera", "Automatic Transmission", "Automatic"),
            ("2006 Porsche 911 Carrera", "Tiptronic", "Tiptronic"),
        ],
    )
    def test_transmission_extraction(self, title: str, details: str, expected: str):
        """Transmission correctly extracted from title or details."""
        assert parse_transmission(title, details) == expected

    def test_title_takes_precedence_over_details(self):
        """If transmission in title, use that even if details differ."""
        # Title says PDK, details say manual - title wins
        result = parse_transmission("2022 Porsche 911 PDK", "Manual listed somewhere")
        assert result == "PDK"


class TestParseYear:
    """Test model year extraction from titles."""

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("2022 Porsche 911 Carrera 4S", 2022),
            ("1999 Porsche 911 Carrera", 1999),
            ("2024 Porsche 911 GT3 RS", 2024),
            ("No Year Porsche 911", None),
            # 1970s air-cooled
            ("1974 Porsche 911 Carrera", 1974),
            ("1976 Porsche 930 Turbo Carrera Project", 1976),
            ("Twin-Turbocharged 3.0L-Powered 1974 Porsche 911 Cabriolet", 1974),
            # Edge cases
            ("1963 Porsche 911", 1963),
            ("1989 Porsche 911 Carrera Coupe G50", 1989),
        ],
    )
    def test_year_extraction(self, title: str, expected: int | None):
        """Year correctly extracted from title."""
        assert parse_year(title) == expected


class TestParsePrice:
    """Test price parsing from BaT format."""

    @pytest.mark.parametrize(
        "price_text,expected",
        [
            ("$125,000", 125000),
            ("$85,500", 85500),
            ("$1,250,000", 1250000),
            ("Sold for $135,000", 135000),
            ("", None),
            (None, None),
        ],
    )
    def test_price_parsing(self, price_text: str | None, expected: int | None):
        """Price correctly parsed from various formats."""
        assert parse_price(price_text) == expected


class TestParseMileage:
    """Test mileage parsing from BaT format."""

    @pytest.mark.parametrize(
        "mileage_text,expected",
        [
            ("15,000 Miles", 15000),
            ("8,500 miles", 8500),
            ("125000 mi", 125000),
            ("3,200 Miles Shown", 3200),
            ("", None),
            (None, None),
        ],
    )
    def test_mileage_parsing(self, mileage_text: str | None, expected: int | None):
        """Mileage correctly parsed from various formats."""
        assert parse_mileage(mileage_text) == expected


class TestIsNonCarListing:
    """Test non-car listing detection (parts, wheels, literature).

    Note: We use ULTRA conservative keyword filtering to avoid false positives.
    Most parts listings are caught by the $10k price floor in validate_scraped_data().
    """

    @pytest.mark.parametrize(
        "card_text,title,expected",
        [
            # Non-car listings caught by keywords (should return True)
            ("Porsche Literature", "Collection of Porsche Literature & Automobilia", True),
            ("Porsche Transaxle", "Porsche 930 4-Speed Transaxle", True),
            # Wheels are NOT filtered by keywords (too risky - GT3 RS has center-lock wheels)
            # These get filtered by $10k price floor instead
            ("Center-Lock Wheels for Porsche", "20×9″ and 20×12″ Center-Lock Wheels for Porsche 991 GT3", False),
            # Actual car listings (should return False)
            ("2020 Porsche 911 Carrera 4S Coupe", "2020 Porsche 911 Carrera 4S Coupe", False),
            ("2019 Porsche 911 GT3 RS Weissach", "2019 Porsche 911 GT3 RS Weissach", False),
            # Edge cases that should NOT be filtered (common car description terms)
            ("2020 Porsche 911 with limited-slip differential", "2020 Porsche 911 Carrera S", False),
            ("1989 Porsche 911 with rebuilt gearbox", "1989 Porsche 911 Carrera", False),
            ("2020 Porsche 911 with Sport Exhaust", "2020 Porsche 911 Carrera S", False),
            ("Modified 1984 Porsche 911 6-Speed Manual Transmission", "Modified 1984 Porsche 911 Carrera Coupe", False),
        ],
    )
    def test_non_car_detection(self, card_text: str, title: str, expected: bool):
        """Non-car listings correctly identified."""
        assert is_non_car_listing(card_text, title) == expected
