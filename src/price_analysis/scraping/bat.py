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
    match = re.search(r"\b(19[89]\d|20[0-2]\d)\b", title)
    if match:
        return int(match.group(1))
    return None


def parse_price(price_text: str) -> int | None:
    """Parse price from BaT format (e.g., '$125,000').

    Args:
        price_text: Raw price string

    Returns:
        Price as int or None
    """
    if not price_text:
        return None
    # Remove currency symbols, commas, and whitespace
    cleaned = re.sub(r"[^\d]", "", price_text)
    if cleaned:
        return int(cleaned)
    return None


def parse_mileage(mileage_text: str) -> int | None:
    """Parse mileage from BaT format (e.g., '15,000 Miles').

    Args:
        mileage_text: Raw mileage string

    Returns:
        Mileage as int or None
    """
    if not mileage_text:
        return None
    match = re.search(r"([\d,]+)\s*(?:Miles?|mi)", mileage_text, re.IGNORECASE)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def fetch_search_results(
    driver: webdriver.Chrome,
    query: str,
    max_pages: int = 20,
    delay: float = 2.5,
) -> list[str]:
    """Fetch listing URLs from BaT search results.

    Args:
        driver: Selenium WebDriver
        query: Search query (e.g., "Porsche 911")
        max_pages: Maximum pages to scrape
        delay: Delay between page requests (seconds)

    Returns:
        List of listing URLs
    """
    base_url = "https://bringatrailer.com/auctions/results/"
    listing_urls: list[str] = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?s={query.replace(' ', '+')}&page={page}"
        logger.info(f"Fetching search page {page}: {url}")

        driver.get(url)
        time.sleep(delay)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "listing-card"))
            )
        except Exception:
            logger.warning(f"No results found on page {page}, stopping pagination")
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find listing cards - adjust selector based on actual BaT structure
        cards = soup.find_all("a", class_="listing-card")
        if not cards:
            # Try alternative selectors
            cards = soup.find_all("a", href=re.compile(r"/listing/"))

        if not cards:
            logger.warning(f"No listing cards found on page {page}")
            break

        for card in cards:
            href = card.get("href", "")
            if href and "/listing/" in href:
                if not href.startswith("http"):
                    href = f"https://bringatrailer.com{href}"
                listing_urls.append(href)

        logger.info(f"Found {len(cards)} listings on page {page}")

        # Check if there's a next page
        next_button = soup.find("a", class_="next")
        if not next_button:
            logger.info("No more pages available")
            break

    return list(set(listing_urls))  # Deduplicate


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

        # Extract title
        title_elem = soup.find("h1", class_="post-title")
        if not title_elem:
            title_elem = soup.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Extract sale price (for completed auctions)
        price = None
        price_elem = soup.find("span", class_="info-value")
        if price_elem:
            price = parse_price(price_elem.get_text())

        # Alternative: look for "Sold for" text
        if price is None:
            sold_text = soup.find(string=re.compile(r"Sold\s+for", re.IGNORECASE))
            if sold_text:
                price = parse_price(sold_text)

        # Extract essentials block for details
        essentials = soup.find("div", class_="essentials")
        details_text = essentials.get_text() if essentials else ""

        # Extract specific fields from essentials
        mileage = None
        color = None
        location = None

        # Look for labeled items in essentials
        items = soup.find_all("li", class_="essential-item") or soup.find_all(
            "div", class_="item"
        )
        for item in items:
            text = item.get_text(strip=True).lower()
            if "mile" in text:
                mileage = parse_mileage(item.get_text())
            elif "color" in text or any(c in text for c in ["black", "white", "silver", "red"]):
                color = item.get_text(strip=True)
            elif any(loc in text for loc in ["california", "texas", "florida", "new york"]):
                location = item.get_text(strip=True)

        # Extract sale date
        sale_date = None
        date_elem = soup.find("span", class_="date")
        if date_elem:
            try:
                date_text = date_elem.get_text(strip=True)
                sale_date = datetime.strptime(date_text, "%B %d, %Y")
            except ValueError:
                pass

        # Parse derived fields
        year = parse_year(title)
        generation = parse_generation(title, year) if year else None
        trim = parse_trim(title)
        transmission = parse_transmission(title, details_text)

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

    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def fetch_auctions(
    query: str = "Porsche 911",
    max_pages: int = 20,
    delay: float = 2.5,
    headless: bool = True,
) -> list[AuctionListing]:
    """Scrape completed BaT auctions for a search query.

    Main entry point for scraping. Creates driver, fetches search results,
    then fetches details for each listing.

    Args:
        query: Search term (e.g., "Porsche 911 992")
        max_pages: Maximum search result pages to scrape
        delay: Delay between requests (seconds) - be polite!
        headless: Run browser without GUI

    Returns:
        List of AuctionListing objects
    """
    driver = create_driver(headless=headless)
    listings: list[AuctionListing] = []

    try:
        # Get listing URLs from search
        urls = fetch_search_results(driver, query, max_pages, delay)
        logger.info(f"Found {len(urls)} unique listings to fetch")

        # Fetch each listing's details
        for i, url in enumerate(urls):
            logger.info(f"Processing listing {i + 1}/{len(urls)}")
            listing = fetch_listing_details(driver, url, delay)
            if listing:
                listings.append(listing)
            time.sleep(delay)  # Extra politeness

    finally:
        driver.quit()

    logger.info(f"Successfully scraped {len(listings)} listings")
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
