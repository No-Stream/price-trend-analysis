"""Tests for BaT scraping module using saved HTML fixtures.

These tests verify parsing against actual saved HTML pages, providing
regression tests if BaT changes their DOM structure.
"""

from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from price_analysis.scraping.bat import (
    AuctionListing,
    DataQualityError,
    parse_generation,
    parse_listing_from_soup,
    parse_mileage,
    parse_price,
    parse_transmission,
    parse_trim,
    parse_year,
    validate_scraped_data,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def listing_992_html() -> str:
    """Load saved 992.1 listing HTML."""
    return (FIXTURES_DIR / "bat_listing_992.html").read_text()


@pytest.fixture
def listing_997_html() -> str:
    """Load saved 997.2 listing HTML."""
    return (FIXTURES_DIR / "bat_listing_997.html").read_text()


@pytest.fixture
def search_page_html() -> str:
    """Load saved search results HTML."""
    return (FIXTURES_DIR / "bat_search_page.html").read_text()


class TestParseFunctions:
    """Test individual parsing functions with known inputs."""

    def test_parse_price_from_dollar_string(self):
        """Parse price from typical BaT format."""
        assert parse_price("$137,000") == 137000
        assert parse_price("USD $137,000") == 137000
        assert parse_price("$52,500") == 52500

    def test_parse_mileage_from_text(self):
        """Parse mileage from various formats."""
        assert parse_mileage("6,000 Miles") == 6000
        assert parse_mileage("6k Miles") == 6000
        assert parse_mileage("15,000 Miles Shown") == 15000
        assert parse_mileage("100 Miles") == 100

    def test_parse_year_from_title(self):
        """Extract model year from listing title."""
        assert parse_year("2020 Porsche 911 Carrera 4S Coupe") == 2020
        assert parse_year("6k-Mile 2020 Porsche 911") == 2020
        assert parse_year("1995 Porsche 911 Carrera") == 1995

    def test_parse_generation_from_year(self):
        """Map model year to generation code."""
        assert parse_generation("", 2020) == "992.1"
        assert parse_generation("", 2022) == "992.1"
        assert parse_generation("", 2009) == "997.2"
        assert parse_generation("", 2017) == "991.2"
        assert parse_generation("", 2000) == "996.1"

    def test_parse_generation_returns_none_for_aircooled(self):
        """Air-cooled cars (pre-996) should return None."""
        assert parse_generation("", 1995) is None
        assert parse_generation("", 1989) is None
        assert parse_generation("", 1976) is None

    def test_parse_trim_from_title(self):
        """Extract trim level from listing title."""
        assert parse_trim("2020 Porsche 911 Carrera 4S Coupe") == "Carrera 4S"
        assert parse_trim("2019 Porsche 911 GT3 RS Weissach") == "GT3 RS"
        assert parse_trim("2017 Porsche 911 Turbo S") == "Turbo S"
        assert parse_trim("2022 Porsche 911 Targa 4") == "Targa 4"

    def test_parse_transmission_from_title(self):
        """Extract transmission from title text."""
        assert parse_transmission("2020 Porsche 911 Carrera 4S PDK") == "PDK"
        assert parse_transmission("2020 Porsche 911 Carrera S 6-Speed Manual") == "Manual"
        assert parse_transmission("", "Seven-Speed Manual Transmission") == "Manual"


@pytest.mark.slow
class TestListingPageParsing:
    """Tests that parse actual saved HTML listing pages.

    These tests verify end-to-end parsing against real BaT page structure.
    Marked slow since they're integration tests (though reading from disk is fast).
    """

    def test_extract_title_from_992_listing(self, listing_992_html):
        """Title extracted correctly from page."""
        soup = BeautifulSoup(listing_992_html, "html.parser")

        # Title is in <title> tag or h1
        title_tag = soup.find("title")
        assert title_tag is not None
        title_text = title_tag.get_text()

        assert "2020 Porsche 911 Carrera 4S" in title_text

    def test_extract_price_from_992_listing(self, listing_992_html):
        """Price extracted correctly from page."""
        soup = BeautifulSoup(listing_992_html, "html.parser")

        # Price is in span.info-value with "Sold for" text
        info_value = soup.select_one("span.info-value")
        assert info_value is not None

        text = info_value.get_text()
        assert "$137,000" in text

        # Parse the price
        price = parse_price(text)
        assert price == 137000

    def test_extract_date_from_992_listing(self, listing_992_html):
        """Sale date extracted correctly from page."""
        soup = BeautifulSoup(listing_992_html, "html.parser")

        # Date is in span.date within info-value
        date_span = soup.select_one("span.info-value span.date")
        assert date_span is not None

        date_text = date_span.get_text()
        assert "12/24/25" in date_text

    def test_extract_essentials_from_992_listing(self, listing_992_html):
        """Essentials section contains expected data."""
        soup = BeautifulSoup(listing_992_html, "html.parser")

        essentials = soup.select_one("div.essentials")
        assert essentials is not None

        essentials_text = essentials.get_text()

        # Check for expected content
        assert "Phoenix, Arizona" in essentials_text  # Location
        assert "Miles" in essentials_text  # Mileage indicator
        assert "PDK" in essentials_text or "Manual" in essentials_text  # Transmission
        assert "Green" in essentials_text or "Paint" in essentials_text  # Color

    def test_extract_mileage_from_992_listing(self, listing_992_html):
        """Mileage extracted from essentials section."""
        soup = BeautifulSoup(listing_992_html, "html.parser")

        essentials = soup.select_one("div.essentials")
        assert essentials is not None

        # Find list items containing mileage
        items = essentials.select("li")
        mileage_item = None
        for item in items:
            text = item.get_text()
            if "Miles" in text:
                mileage_item = text
                break

        assert mileage_item is not None
        # Should be "6k Miles" for this listing
        assert "6k" in mileage_item or "6,000" in mileage_item

    def test_extract_location_from_992_listing(self, listing_992_html):
        """Location extracted from essentials section."""
        soup = BeautifulSoup(listing_992_html, "html.parser")

        essentials = soup.select_one("div.essentials")
        assert essentials is not None

        # Location follows "Location" strong tag
        text = essentials.get_text()
        assert "Phoenix" in text
        assert "Arizona" in text


@pytest.mark.slow
class TestSearchPageParsing:
    """Tests that parse actual saved HTML search pages."""

    def test_search_page_has_listings(self, search_page_html):
        """Search page contains listing links."""
        soup = BeautifulSoup(search_page_html, "html.parser")

        # Look for links to listings
        listing_links = soup.select('a[href*="/listing/"]')
        assert len(listing_links) > 0

    def test_search_page_listing_count(self, search_page_html):
        """Search page has reasonable number of listings."""
        soup = BeautifulSoup(search_page_html, "html.parser")

        listing_links = soup.select('a[href*="/listing/"]')
        # Deduplicate URLs
        urls = {link.get("href") for link in listing_links if link.get("href")}

        # Should have multiple listings on search page
        assert len(urls) >= 5


@pytest.mark.slow
class TestFullListingParsing:
    """Full integration tests using parse_listing_from_soup().

    These tests verify the complete parsing flow produces correct AuctionListing
    dataclass output from saved HTML fixtures.
    """

    def test_parse_992_listing_returns_auction_listing(self, listing_992_html):
        """parse_listing_from_soup returns AuctionListing dataclass."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup, "https://example.com/test")
        assert isinstance(listing, AuctionListing)

    def test_parse_992_listing_title(self, listing_992_html):
        """Title extracted correctly."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert "2020 Porsche 911 Carrera 4S" in listing.title_raw

    def test_parse_992_listing_price(self, listing_992_html):
        """Price extracted correctly as integer."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.sale_price == 137000

    def test_parse_992_listing_sale_date(self, listing_992_html):
        """Sale date parsed correctly."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.sale_date is not None
        assert listing.sale_date.year == 2025
        assert listing.sale_date.month == 12
        assert listing.sale_date.day == 24

    def test_parse_992_listing_model_year(self, listing_992_html):
        """Model year extracted from title."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.model_year == 2020

    def test_parse_992_listing_generation(self, listing_992_html):
        """Generation derived from model year."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.generation == "992.1"

    def test_parse_992_listing_trim(self, listing_992_html):
        """Trim extracted from title."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.trim == "Carrera 4S"

    def test_parse_992_listing_transmission(self, listing_992_html):
        """Transmission extracted from title/essentials."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.transmission == "PDK"

    def test_parse_992_listing_mileage(self, listing_992_html):
        """Mileage extracted from essentials section."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.mileage == 6000

    def test_parse_992_listing_location(self, listing_992_html):
        """Location extracted correctly (city, state only)."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.location is not None
        assert "Phoenix" in listing.location
        # Should NOT contain the entire essentials section
        assert "BaT Essentials" not in listing.location
        assert "Seller" not in listing.location

    def test_parse_992_listing_color(self, listing_992_html):
        """Color extracted from essentials section."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.color is not None
        assert "Green" in listing.color or "Aventurine" in listing.color

    def test_parse_992_listing_all_required_fields_populated(self, listing_992_html):
        """All required fields for modeling are populated."""
        soup = BeautifulSoup(listing_992_html, "html.parser")
        listing = parse_listing_from_soup(soup, "https://bat.com/listing/123")

        # Required fields per AGENTS.md
        assert listing.listing_url == "https://bat.com/listing/123"
        assert listing.sale_price is not None
        assert listing.sale_date is not None
        assert listing.model_year is not None
        assert listing.generation is not None
        assert listing.trim is not None
        assert listing.transmission is not None
        assert listing.mileage is not None

    def test_parse_997_listing_generation(self, listing_997_html):
        """997.2 generation derived correctly from 2009 model year."""
        soup = BeautifulSoup(listing_997_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.model_year == 2009
        assert listing.generation == "997.2"

    def test_parse_997_listing_transmission(self, listing_997_html):
        """997 listing with 6-speed is recognized as Manual."""
        soup = BeautifulSoup(listing_997_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.transmission == "Manual"

    def test_parse_997_listing_trim(self, listing_997_html):
        """997 trim extracted correctly."""
        soup = BeautifulSoup(listing_997_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        assert listing.trim == "Carrera 4S"

    def test_parse_997_listing_price_is_reasonable(self, listing_997_html):
        """997.2 C4S 6-speed price should be in expected range."""
        soup = BeautifulSoup(listing_997_html, "html.parser")
        listing = parse_listing_from_soup(soup)
        # 997.2 C4S 6-speed typically $50k-$100k
        assert listing.sale_price is not None
        assert 40000 < listing.sale_price < 150000


class TestDataQualityValidation:
    """Tests for validate_scraped_data() function."""

    def test_valid_data_passes(self):
        """Good data passes validation."""
        df = pd.DataFrame(
            {
                "listing_url": ["url1", "url2"],
                "sale_price": [100000, 120000],
                "model_year": [2020, 2021],
            }
        )
        # Should not raise
        validate_scraped_data(df)

    def test_empty_dataframe_raises(self):
        """Empty DataFrame raises DataQualityError."""
        df = pd.DataFrame()
        with pytest.raises(DataQualityError, match="empty"):
            validate_scraped_data(df)

    def test_all_missing_column_raises(self):
        """Column that is 100% missing raises error."""
        df = pd.DataFrame(
            {
                "listing_url": ["url1", "url2"],
                "sale_price": [100000, 120000],
                "color": [None, None],  # All missing
            }
        )
        with pytest.raises(DataQualityError, match="color.*100% missing"):
            validate_scraped_data(df)

    def test_price_missing_above_threshold_raises(self):
        """Price missing rate above threshold raises error."""
        df = pd.DataFrame(
            {
                "listing_url": ["url1", "url2", "url3", "url4", "url5"],
                "sale_price": [100000, None, None, None, None],  # 80% missing
            }
        )
        with pytest.raises(DataQualityError, match="sale_price missing rate"):
            validate_scraped_data(df, max_price_missing_pct=0.10)

    def test_price_missing_below_threshold_passes(self):
        """Price missing rate below threshold passes."""
        df = pd.DataFrame(
            {
                "listing_url": ["url1", "url2", "url3", "url4", "url5"],
                "sale_price": [100000, 120000, 130000, 140000, None],  # 20% missing
            }
        )
        # 20% missing is above 10% default, so should fail
        with pytest.raises(DataQualityError):
            validate_scraped_data(df, max_price_missing_pct=0.10)

        # But passes with higher threshold
        validate_scraped_data(df, max_price_missing_pct=0.25)
