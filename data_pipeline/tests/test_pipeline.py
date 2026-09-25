
import os
import sys
import sqlite3

import pandas as pd

# Allow imports from data_pipeline/src
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
)

from scraper import scrape_books
from cleaner import clean_books
from validator import validate_books


def test_scraper_returns_at_least_60_books():
    """Scraper should collect at least 60 books."""

    df = scrape_books(num_pages=5)

    assert len(df) >= 60
    assert "title" in df.columns
    assert "price" in df.columns
    assert "star_rating" in df.columns
    assert "availability" in df.columns
    assert "category" in df.columns


def test_cleaner_creates_required_columns():
    """Cleaner should create correctly named columns."""

    raw_df = pd.DataFrame({
        "title": ["Test Book"],
        "price": ["£10.00"],
        "star_rating": ["Five"],
        "availability": ["In stock"],
        "category": ["Books"]
    })

    cleaned_df = clean_books(raw_df)

    expected_columns = [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category"
    ]

    assert list(cleaned_df.columns) == expected_columns
    assert cleaned_df.loc[0, "price_gbp"] == 10.00
    assert cleaned_df.loc[0, "price_inr"] == 1055.00
    assert cleaned_df.loc[0, "rating"] == 5
    assert bool(cleaned_df.loc[0, "in_stock"]) is True


def test_validator_accepts_clean_data():
    """Validator should accept valid cleaned data."""

    raw_df = pd.DataFrame({
        "title": ["Test Book"] * 60,
        "price": ["£10.00"] * 60,
        "star_rating": ["Five"] * 60,
        "availability": ["In stock"] * 60,
        "category": ["Books"] * 60
    })

    cleaned_df = clean_books(raw_df)

    assert validate_books(cleaned_df) is True


def test_database_has_required_tables():
    """SQLite database should contain categories and books tables."""

    db_path = "data_pipeline/data/zepto_books.db"

    assert os.path.exists(db_path)

    conn = sqlite3.connect(db_path)

    tables = pd.read_sql(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        """,
        conn
    )

    table_names = set(tables["name"])

    assert "categories" in table_names
    assert "books" in table_names

    conn.close()


if __name__ == "__main__":
    print("Running pipeline tests...")

    test_scraper_returns_at_least_60_books()
    print("✓ Scraper test passed")

    test_cleaner_creates_required_columns()
    print("✓ Cleaner test passed")

    test_validator_accepts_clean_data()
    print("✓ Validator test passed")

    test_database_has_required_tables()
    print("✓ Database test passed")

    print("\n===== ALL TESTS PASSED =====")
