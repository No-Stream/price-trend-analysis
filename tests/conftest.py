"""Shared test fixtures for price analysis tests."""

from datetime import datetime

import pandas as pd
import pytest


@pytest.fixture
def sample_listings_df() -> pd.DataFrame:
    """Realistic sample of BaT-style listings for testing.

    Includes:
    - Multiple generations, trims, transmissions
    - Some missing fields
    - Edge cases (high mileage, unusual prices)
    """
    data = [
        # Complete records - various generations
        {
            "listing_url": "https://bringatrailer.com/listing/2022-porsche-911-carrera-4s-1",
            "title_raw": "2022 Porsche 911 Carrera 4S",
            "sale_price": 135000,
            "sale_date": datetime(2024, 6, 15),
            "model_year": 2022,
            "generation": "992.1",
            "trim": "Carrera 4S",
            "transmission": "PDK",
            "mileage": 12000,
            "color": "GT Silver Metallic",
            "location": "California",
        },
        {
            "listing_url": "https://bringatrailer.com/listing/2021-porsche-911-carrera-s-1",
            "title_raw": "2021 Porsche 911 Carrera S",
            "sale_price": 118000,
            "sale_date": datetime(2024, 5, 20),
            "model_year": 2021,
            "generation": "992.1",
            "trim": "Carrera S",
            "transmission": "Manual",
            "mileage": 8500,
            "color": "Black",
            "location": "Texas",
        },
        {
            "listing_url": "https://bringatrailer.com/listing/2019-porsche-911-gt3-1",
            "title_raw": "2019 Porsche 911 GT3",
            "sale_price": 185000,
            "sale_date": datetime(2024, 4, 10),
            "model_year": 2019,
            "generation": "991.2",
            "trim": "GT3",
            "transmission": "Manual",
            "mileage": 5200,
            "color": "Miami Blue",
            "location": "Florida",
        },
        {
            "listing_url": "https://bringatrailer.com/listing/2017-porsche-911-turbo-s-1",
            "title_raw": "2017 Porsche 911 Turbo S",
            "sale_price": 145000,
            "sale_date": datetime(2024, 3, 5),
            "model_year": 2017,
            "generation": "991.2",
            "trim": "Turbo S",
            "transmission": "PDK",
            "mileage": 22000,
            "color": "White",
            "location": "New York",
        },
        {
            "listing_url": "https://bringatrailer.com/listing/2010-porsche-911-carrera-s-1",
            "title_raw": "2010 Porsche 911 Carrera S",
            "sale_price": 58000,
            "sale_date": datetime(2024, 2, 28),
            "model_year": 2010,
            "generation": "997.2",
            "trim": "Carrera S",
            "transmission": "Manual",
            "mileage": 45000,
            "color": "Guards Red",
            "location": "California",
        },
        {
            "listing_url": "https://bringatrailer.com/listing/2006-porsche-911-carrera-4s-1",
            "title_raw": "2006 Porsche 911 Carrera 4S",
            "sale_price": 52000,
            "sale_date": datetime(2024, 1, 15),
            "model_year": 2006,
            "generation": "997.1",
            "trim": "Carrera 4S",
            "transmission": "Manual",
            "mileage": 68000,
            "color": "Midnight Blue",
            "location": "Illinois",
        },
        # High mileage example
        {
            "listing_url": "https://bringatrailer.com/listing/2015-porsche-911-carrera-1",
            "title_raw": "2015 Porsche 911 Carrera",
            "sale_price": 55000,
            "sale_date": datetime(2023, 12, 10),
            "model_year": 2015,
            "generation": "991.1",
            "trim": "Carrera",
            "transmission": "PDK",
            "mileage": 95000,
            "color": "Black",
            "location": "Arizona",
        },
        # Missing mileage
        {
            "listing_url": "https://bringatrailer.com/listing/2020-porsche-911-carrera-4-1",
            "title_raw": "2020 Porsche 911 Carrera 4",
            "sale_price": 105000,
            "sale_date": datetime(2024, 7, 1),
            "model_year": 2020,
            "generation": "992.1",
            "trim": "Carrera 4",
            "transmission": "PDK",
            "mileage": None,
            "color": "Chalk",
            "location": "Nevada",
        },
        # Missing transmission
        {
            "listing_url": "https://bringatrailer.com/listing/2018-porsche-911-targa-4s-1",
            "title_raw": "2018 Porsche 911 Targa 4S",
            "sale_price": 125000,
            "sale_date": datetime(2024, 6, 5),
            "model_year": 2018,
            "generation": "991.2",
            "trim": "Targa 4S",
            "transmission": None,
            "mileage": 18000,
            "color": "Racing Yellow",
            "location": "Colorado",
        },
        # PTS color
        {
            "listing_url": "https://bringatrailer.com/listing/2023-porsche-911-carrera-gts-1",
            "title_raw": "2023 Porsche 911 Carrera GTS",
            "sale_price": 175000,
            "sale_date": datetime(2024, 7, 20),
            "model_year": 2023,
            "generation": "992.1",
            "trim": "Carrera S",  # GTS parsed as Carrera S (simplification)
            "transmission": "Manual",
            "mileage": 3500,
            "color": "PTS Rubystone Red",
            "location": "California",
        },
        # Invalid record - missing price (should fail validation)
        {
            "listing_url": "https://bringatrailer.com/listing/2021-porsche-911-carrera-4s-invalid",
            "title_raw": "2021 Porsche 911 Carrera 4S",
            "sale_price": None,
            "sale_date": datetime(2024, 8, 1),
            "model_year": 2021,
            "generation": "992.1",
            "trim": "Carrera 4S",
            "transmission": "PDK",
            "mileage": 10000,
            "color": "Black",
            "location": "Texas",
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def minimal_model_data() -> pd.DataFrame:
    """Minimal dataset for model smoke testing.

    - Has required columns in correct format
    - Multiple levels per categorical
    - Enough rows for model to fit (~30)
    """
    import numpy as np

    np.random.seed(42)

    n = 30
    generations = ["991.2", "992.1"]
    trims = ["Carrera", "Carrera S", "Carrera 4S"]
    transmissions = ["PDK", "Manual"]

    data = {
        "log_price": np.random.normal(11.7, 0.3, n),  # ~$120k median
        "age": np.random.randint(1, 8, n),
        "mileage_scaled": np.random.normal(0, 1, n),
        "is_low_mileage": np.random.choice([0, 1], n, p=[0.85, 0.15]),  # ~15% low mileage
        "sale_year": np.random.choice([2023, 2024], n),
        "generation": pd.Categorical(np.random.choice(generations, n)),
        "trim": pd.Categorical(np.random.choice(trims, n)),
        "transmission": pd.Categorical(np.random.choice(transmissions, n)),
    }
    return pd.DataFrame(data)
