"""Bring a Trailer (BaT) scraper for Porsche 911 auction data.

Adapted from KaledDahleh/bring-a-trailer-tracker approach.
Uses Selenium for JS rendering since BaT requires it.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger: logging.Logger = logging.getLogger(__name__)


def save_debug_page(
    driver: webdriver.Chrome, filename: str, output_dir: str = "tests/fixtures"
) -> str:
    """Save current page source for debugging selectors.

    Args:
        driver: Selenium WebDriver with page loaded
        filename: Output filename (without .html extension)
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{filename}.html"
    filepath.write_text(driver.page_source)
    logger.info(f"Saved debug page to {filepath}")
    return str(filepath)


# Generation mapping by model year for 911 (water-cooled era)
# Note: Some years overlap between generations; title parsing helps disambiguate
GENERATION_BY_YEAR: dict[tuple[int, int], str] = {
    (1999, 2001): "996.1",
    (2002, 2004): "996.2",
    (2005, 2008): "997.1",
    (2009, 2012): "997.2",
    (2012, 2015): "991.1",  # 2012 overlaps with 997.2
    (2016, 2019): "991.2",
    (2020, 2023): "992.1",
    (2024, 2030): "992.2",
}

# Trim patterns to search for in titles (order matters - more specific first)
TRIM_PATTERNS: list[tuple[str, str]] = [
    (r"\bGT2\s*RS\b", "GT2 RS"),
    (r"\bGT3\s*RS\b", "GT3 RS"),
    (r"\bGT3\s*Touring\b", "GT3 Touring"),
    (r"\bGT3\b", "GT3"),
    (r"\bTurbo\s*S\b", "Turbo S"),
    (r"\bTurbo\b", "Turbo"),
    (r"\bTarga\s*4S\b", "Targa 4S"),
    (r"\bTarga\s*4\b", "Targa 4"),
    (r"\bTarga\b", "Targa"),
    (r"\bCarrera\s*4S\b", "Carrera 4S"),
    (r"\bCarrera\s*S\b", "Carrera S"),
    (r"\bCarrera\s*4\b", "Carrera 4"),
    (r"\bCarrera\b", "Carrera"),
    (r"\bC4S\b", "Carrera 4S"),
    (r"\bC2S\b", "Carrera S"),
    (r"\bC4\b", "Carrera 4"),
    (r"\bC2\b", "Carrera"),
]

# Transmission patterns
TRANSMISSION_PATTERNS: list[tuple[str, str]] = [
    (r"\bPDK\b", "PDK"),
    (r"\b6-Speed\s*Manual\b", "Manual"),
    (r"\b7-Speed\s*Manual\b", "Manual"),
    (r"\bManual\b", "Manual"),
    (r"\bTiptronic\b", "Tiptronic"),
    (r"\bAutomatic\b", "Automatic"),
]

# Keywords indicating non-car listings (parts, accessories, literature)
# Be ULTRA conservative - only include terms that NEVER appear in car listings
# When in doubt, leave it out - we filter by price downstream ($10k floor)
NON_CAR_KEYWORDS: list[str] = [
    "literature",
    "automobilia",
    "transaxle",  # Parts term - cars say "transmission" not "transaxle"
]

# No title patterns - too risky for false positives
# e.g., "center-lock wheels" could describe a GT3 RS that HAS center-lock wheels
NON_CAR_TITLE_PATTERNS: list[str] = []


@dataclass
class AuctionListing:
    """Represents a single BaT auction listing."""

    listing_url: str
    title_raw: str
    sale_price: int | None
    sale_date: datetime | None
    model_year: int | None
    generation: str | None
    trim: str | None
    transmission: str | None
    mileage: int | None
    color: str | None
    location: str | None


def create_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a Selenium Chrome driver.

    Args:
        headless: Run browser in headless mode (no GUI)

    Returns:
        Configured Chrome WebDriver
    """
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def parse_generation(title: str, year: int) -> str | None:
    """Extract 911 generation from title and model year.

    Uses model year as primary signal, with title parsing for disambiguation
    in overlap years (e.g., 2012 could be 997.2 or 991.1).

    Args:
        title: Raw listing title
        year: Model year of the car

    Returns:
        Generation string (e.g., "992.1") or None if not determinable
    """
    # Check for explicit generation mentions in title
    gen_patterns = [
        (r"\b996\.1\b", "996.1"),
        (r"\b996\.2\b", "996.2"),
        (r"\b997\.1\b", "997.1"),
        (r"\b997\.2\b", "997.2"),
        (r"\b991\.1\b", "991.1"),
        (r"\b991\.2\b", "991.2"),
        (r"\b992\.1\b", "992.1"),
        (r"\b992\.2\b", "992.2"),
    ]

    for pattern, gen in gen_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return gen

    # Fall back to year-based lookup
    for (start, end), gen in GENERATION_BY_YEAR.items():
        if start <= year <= end:
            return gen

    return None


def parse_trim(title: str) -> str | None:
    """Extract trim level from listing title.

    Args:
        title: Raw listing title

    Returns:
        Trim string (e.g., "Carrera 4S") or None if not found
    """
    for pattern, trim in TRIM_PATTERNS:
        if re.search(pattern, title, re.IGNORECASE):
            return trim
    return None


def parse_transmission(title: str, details_text: str = "") -> str | None:
    """Extract transmission type from title or details.

    Args:
        title: Raw listing title
        details_text: Additional details text from listing page

    Returns:
        Transmission string (e.g., "PDK", "Manual") or None
    """
    combined = f"{title} {details_text}"
    for pattern, trans in TRANSMISSION_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return trans
    return None


def parse_year(title: str) -> int | None:
    """Extract model year from title.

    Args:
        title: Raw listing title (typically starts with year)

    Returns:
        Model year as int or None
    """
    # Match years 1960-2029 (covers all 911 generations from original to future)
    match = re.search(r"\b(19[6-9]\d|20[0-2]\d)\b", title)
    if match:
        return int(match.group(1))
    return None


def is_non_car_listing(card_text: str, title: str = "") -> bool:
    """Check if listing text indicates a non-car item (parts, wheels, etc.).

    Args:
        card_text: Full card text from search results
        title: Listing title (if available separately)

    Returns:
        True if this appears to be a non-car listing
    """
    text_lower = card_text.lower()
    title_lower = title.lower() if title else text_lower

    # Check for definitive non-car keywords (these rarely appear in car listings)
    if any(keyword in text_lower for keyword in NON_CAR_KEYWORDS):
        return True

    # Check title-specific patterns (wheels, parts, etc.)
    if any(pattern in title_lower for pattern in NON_CAR_TITLE_PATTERNS):
        return True

    return False


def parse_price(price_text: str) -> int | None:
    """Parse price from BaT format (e.g., '$125,000', 'USD $137,000').

    Extracts only the dollar amount, ignoring any surrounding text like dates.

    Args:
        price_text: Raw price string (may contain other text)

    Returns:
        Price as int or None
    """
    if not price_text:
        return None

    # Look for dollar amount pattern: $123,456 or USD $123,456
    match = re.search(r"\$[\d,]+", price_text)
    if match:
        # Remove $ and commas
        cleaned = match.group().replace("$", "").replace(",", "")
        return int(cleaned)
    return None


def parse_sale_date(date_text: str) -> datetime | None:
    """Parse sale date from BaT format (e.g., 'on 12/24/25', 'December 24, 2025').

    Args:
        date_text: Raw date string

    Returns:
        datetime or None
    """
    if not date_text:
        return None

    # Try MM/DD/YY format (e.g., "12/24/25")
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", date_text)
    if match:
        month, day, year = match.groups()
        year = int(year)
        if year < 100:
            year += 2000  # Convert 25 -> 2025
        return datetime(year, int(month), int(day))

    # Try "Month DD, YYYY" format
    try:
        from dateutil import parser as date_parser

        return date_parser.parse(date_text)
    except Exception:
        pass

    return None


def parse_mileage(mileage_text: str) -> int | None:
    """Parse mileage from BaT format (e.g., '15,000 Miles', '6k Miles').

    Handles both numeric formats (15,000) and shorthand (6k, 100k).

    Args:
        mileage_text: Raw mileage string

    Returns:
        Mileage as int or None
    """
    if not mileage_text:
        return None

    # Handle shorthand like "6k Miles", "100k Miles"
    shorthand_match = re.search(r"(\d+)k\s*(?:Miles?|mi)", mileage_text, re.IGNORECASE)
    if shorthand_match:
        return int(shorthand_match.group(1)) * 1000

    # Handle full format like "15,000 Miles"
    full_match = re.search(r"([\d,]+)\s*(?:Miles?|mi)", mileage_text, re.IGNORECASE)
    if full_match:
        return int(full_match.group(1).replace(",", ""))

    return None


def fetch_search_results(
    driver: webdriver.Chrome,
    query: str,
    max_clicks: int = 50,
    delay: float = 2.5,
    completed_only: bool = True,
) -> list[str]:
    """Fetch listing URLs from BaT search results using Show More button.

    BaT uses infinite scroll with a "Show More" button instead of URL pagination.
    This function clicks the button repeatedly to load all results.

    Args:
        driver: Selenium WebDriver
        query: Search query (e.g., "Porsche 911")
        max_clicks: Maximum times to click "Show More" button
        delay: Delay between clicks (seconds)
        completed_only: If True, only return URLs for completed auctions (skip "Bid to" listings)

    Returns:
        List of listing URLs
    """
    url = f"https://bringatrailer.com/auctions/results/?s={query.replace(' ', '+')}"
    logger.info(f"Fetching search results: {url}")

    driver.get(url)
    time.sleep(delay)

    # Wait for initial listings to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/listing/"]'))
        )
    except Exception:
        logger.warning("No listings found on initial page load")
        return []

    # Click "Show More" repeatedly to load more results
    clicks = 0
    while clicks < max_clicks:
        try:
            # Find and click the Show More button
            show_more = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.button-show-more"))
            )
            driver.execute_script("arguments[0].click();", show_more)
            clicks += 1
            logger.info(f"Clicked 'Show More' ({clicks}/{max_clicks})")
            time.sleep(delay)
        except Exception:
            logger.info("No more 'Show More' button available - reached end of results")
            break

    # Extract listing URLs from the page, optionally filtering to completed auctions
    soup = BeautifulSoup(driver.page_source, "html.parser")
    listing_urls: set[str] = set()
    skipped_active = 0
    skipped_non_car = 0

    # Try to find the completed auctions container first (more reliable filtering)
    completed_container = soup.select_one("#auctions-completed-container, .auctions-completed")
    search_scope = completed_container if completed_container else soup
    if completed_container:
        logger.info("Found completed auctions container - searching within it")
    else:
        logger.info("No completed container found - searching full page")

    # Find listing cards and check their status
    # BaT uses different elements for completed vs active auctions:
    # - Completed: <a class="listing-card bg-white-transparent"> (anchor tag)
    # - Active: <div class="listing-card listing-card-separate"> (div tag)
    cards = search_scope.select("a.listing-card, div.listing-card, li.listing-card, article.listing-card")
    logger.info(f"Found {len(cards)} listing cards on page")

    for card in cards:
        # Get the listing URL - for <a> cards it's the href, for <div> cards we find nested <a>
        if card.name == "a" and card.get("href"):
            href = card.get("href", "")
        else:
            link = card.select_one('a[href*="/listing/"]')
            if not link:
                continue
            href = link.get("href", "")
        if not href or "/listing/" not in href:
            continue

        if not href.startswith("http"):
            href = f"https://bringatrailer.com{href}"

        card_text = card.get_text()

        # Try to extract title from card for more accurate filtering
        title_elem = card.select_one("h3, .listing-card-title, a[href*='/listing/']")
        card_title = title_elem.get_text(strip=True) if title_elem else ""

        # Skip non-car listings (wheels, parts, literature, etc.)
        if is_non_car_listing(card_text, card_title):
            skipped_non_car += 1
            logger.debug(f"Filtered non-car listing: {card_title[:80] if card_title else href}")
            continue

        if completed_only:
            # Check if this is a completed auction by looking for "Sold for" text
            # Active auctions have countdown timers and "Bid:" labels (not "Bid to")
            card_text_lower = card_text.lower()
            if "sold for" in card_text_lower:
                listing_urls.add(href)
            elif "countdown" in str(card) or "bid:" in card_text_lower or "bidding-bid" in str(card):
                # Active auction - has countdown timer or current bid display
                skipped_active += 1
            else:
                # Unknown status - include to be safe
                listing_urls.add(href)
        else:
            listing_urls.add(href)

    # Fallback: if card-based extraction found nothing, try simple link extraction
    if not listing_urls:
        logger.warning("Card-based extraction found no URLs, falling back to link extraction")
        all_links = soup.select('a[href*="/listing/"]')
        logger.warning(f"Fallback found {len(all_links)} total listing links")
        for link in all_links:
            href = link.get("href", "")
            if href and "/listing/" in href:
                if not href.startswith("http"):
                    href = f"https://bringatrailer.com{href}"
                listing_urls.add(href)

    if skipped_non_car > 0:
        logger.info(f"Filtered out {skipped_non_car} non-car listings (parts/wheels/etc)")
    if skipped_active > 0:
        logger.info(f"Filtered out {skipped_active} active auctions from search results")
    logger.info(f"Found {len(listing_urls)} unique listings after {clicks} 'Show More' clicks")
    return list(listing_urls)


def parse_listing_from_soup(soup: BeautifulSoup, url: str = "") -> AuctionListing | None:
    """Parse a BaT listing page from BeautifulSoup object.

    This function extracts all fields from a parsed HTML page.
    Separated from fetch_listing_details() to enable testing with saved HTML.

    Args:
        soup: BeautifulSoup object of listing page
        url: Listing URL (for reference)

    Returns:
        AuctionListing or None if parsing fails
    """
    # Extract title
    title_elem = soup.find("h1", class_="post-title") or soup.find("h1")
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Extract sale price from span.info-value
    price = None
    price_elem = soup.select_one("span.info-value")
    if price_elem:
        price = parse_price(price_elem.get_text())

    # Fallback: look for "Sold for" text
    if price is None:
        sold_text = soup.find(string=re.compile(r"Sold\s+for", re.IGNORECASE))
        if sold_text:
            price = parse_price(sold_text)

    # Extract sale date from span.date within info-value
    sale_date = None
    date_elem = soup.select_one("span.info-value span.date")
    if date_elem:
        sale_date = parse_sale_date(date_elem.get_text())

    # Extract essentials block for details
    essentials = soup.select_one("div.essentials")
    essentials_text = essentials.get_text() if essentials else ""

    # Extract mileage from essentials li items
    mileage = None
    color = None
    location = None

    if essentials:
        for li in essentials.select("li"):
            text = li.get_text(strip=True)
            text_lower = text.lower()

            # Mileage: look for "Miles" in li items
            if "mile" in text_lower and mileage is None:
                mileage = parse_mileage(text)

            # Color: look for "Paint" keyword (e.g., "Aventurine Green Metallic Paint")
            if "paint" in text_lower and color is None:
                # Remove "Paint" suffix
                color = text.replace("Paint", "").replace("paint", "").strip()

        # Location: find the <a> tag after <strong>Location</strong>
        location_label = essentials.find("strong", string=re.compile(r"Location", re.IGNORECASE))
        if location_label:
            # The location is in the next sibling <a> tag
            next_sibling = location_label.find_next_sibling("a")
            if next_sibling:
                location = next_sibling.get_text(strip=True)

    # Parse derived fields from title
    year = parse_year(title)
    generation = parse_generation(title, year) if year else None
    trim = parse_trim(title)
    transmission = parse_transmission(title, essentials_text)

    return AuctionListing(
        listing_url=url,
        title_raw=title,
        sale_price=price,
        sale_date=sale_date,
        model_year=year,
        generation=generation,
        trim=trim,
        transmission=transmission,
        mileage=mileage,
        color=color,
        location=location,
    )


def fetch_listing_details(
    driver: webdriver.Chrome,
    url: str,
    delay: float = 2.0,
) -> AuctionListing | None:
    """Fetch and parse a single BaT listing page.

    Args:
        driver: Selenium WebDriver
        url: Listing URL
        delay: Delay after loading page (seconds)

    Returns:
        AuctionListing or None if parsing fails
    """
    logger.info(f"Fetching listing: {url}")

    try:
        driver.get(url)
        time.sleep(delay)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        return parse_listing_from_soup(soup, url)

    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def fetch_auctions(
    query: str = "Porsche 911",
    max_clicks: int = 50,
    delay: float = 2.5,
    headless: bool = True,
    completed_only: bool = True,
) -> list[AuctionListing]:
    """Scrape completed BaT auctions for a search query.

    Main entry point for scraping. Creates driver, fetches search results
    using "Show More" button pagination, then fetches details for each listing.

    Args:
        query: Search term (e.g., "Porsche 911 992")
        max_clicks: Maximum times to click "Show More" button
        delay: Delay between requests (seconds) - be polite!
        headless: Run browser without GUI
        completed_only: If True, filter to completed auctions at search level

    Returns:
        List of AuctionListing objects
    """
    driver = create_driver(headless=headless)
    listings: list[AuctionListing] = []
    skipped_active = 0

    try:
        # Get listing URLs from search using Show More pagination
        # Filter to completed auctions at the search results level to avoid unnecessary fetches
        urls = fetch_search_results(driver, query, max_clicks, delay, completed_only=completed_only)
        logger.info(f"Found {len(urls)} unique listings to fetch")

        # Fetch each listing's details
        for i, url in enumerate(urls):
            logger.info(f"Processing listing {i + 1}/{len(urls)}")
            listing = fetch_listing_details(driver, url, delay)
            if listing is None:
                logger.warning(f"Failed to parse listing: {url}")
            elif completed_only and listing.sale_price is None:
                # Secondary check: some listings may slip through if card text was unclear
                skipped_active += 1
                logger.info(f"Skipped (no price - likely active): {listing.title_raw[:60]}...")
            else:
                listings.append(listing)
            time.sleep(delay)  # Extra politeness

    finally:
        driver.quit()

    if skipped_active > 0:
        logger.info(f"Skipped {skipped_active} active auctions in secondary filter")
    logger.info(f"Successfully scraped {len(listings)} completed listings")
    return listings


def listings_to_dataframe(listings: list[AuctionListing]):
    """Convert list of AuctionListing to pandas DataFrame.

    Args:
        listings: List of AuctionListing objects

    Returns:
        pandas DataFrame with one row per listing
    """
    import pandas as pd

    records = [
        {
            "listing_url": lst.listing_url,
            "title_raw": lst.title_raw,
            "sale_price": lst.sale_price,
            "sale_date": lst.sale_date,
            "model_year": lst.model_year,
            "generation": lst.generation,
            "trim": lst.trim,
            "transmission": lst.transmission,
            "mileage": lst.mileage,
            "color": lst.color,
            "location": lst.location,
        }
        for lst in listings
    ]
    return pd.DataFrame(records)


class DataQualityError(Exception):
    """Raised when scraped data fails quality checks."""

    pass


def validate_scraped_data(
    df,
    max_price_missing_pct: float = 0.10,
    min_price: int = 10000,
    required_cols: list[str] | None = None,
):
    """Validate and filter scraped data quality.

    Filters out listings below min_price (likely parts/salvage), then validates.
    Raises DataQualityError if checks fail.

    Args:
        df: DataFrame from listings_to_dataframe()
        max_price_missing_pct: Maximum allowed missing rate for sale_price (default 10%)
        min_price: Minimum price to keep (default $10,000 - filters parts/salvage)
        required_cols: Columns that cannot be 100% missing (default: all columns)

    Returns:
        Filtered DataFrame with low-price listings removed

    Raises:
        DataQualityError: If any validation check fails
    """
    if len(df) == 0:
        raise DataQualityError("No listings scraped - DataFrame is empty")

    # Filter out listings below min_price (likely parts, salvage, or non-cars)
    if "sale_price" in df.columns and min_price > 0:
        low_price_mask = df["sale_price"] < min_price
        n_filtered = low_price_mask.sum()
        if n_filtered > 0:
            logger.info(f"Filtered {n_filtered} listings below ${min_price:,} (likely parts/salvage)")
            df = df[~low_price_mask].copy()

    if len(df) == 0:
        raise DataQualityError(f"No listings remaining after filtering below ${min_price:,}")

    errors: list[str] = []

    # Check for columns that are entirely missing
    if required_cols is None:
        required_cols = list(df.columns)

    for col in required_cols:
        if col in df.columns and df[col].isna().all():
            errors.append(f"Column '{col}' is 100% missing")

    # Check price missing rate
    if "sale_price" in df.columns:
        price_missing_pct = df["sale_price"].isna().mean()
        if price_missing_pct > max_price_missing_pct:
            errors.append(
                f"sale_price missing rate {price_missing_pct:.1%} exceeds threshold {max_price_missing_pct:.1%}"
            )

    if errors:
        raise DataQualityError("Data quality checks failed:\n  - " + "\n  - ".join(errors))

    logger.info(
        f"Data quality checks passed: {len(df)} listings, {df['sale_price'].notna().sum()} with prices"
    )

    return df
