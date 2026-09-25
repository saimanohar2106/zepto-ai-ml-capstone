
import os
import sqlite3

import pandas as pd

from scraper import scrape_books
from cleaner import clean_books
from validator import validate_books


DB_NAME = "data_pipeline/data/zepto_books.db"
RAW_FILE = "data_pipeline/data/raw_books.csv"
CLEANED_FILE = "data_pipeline/data/cleaned_books.csv"
OUTPUT_DIR = "data_pipeline/outputs"


QUERIES = {
    "Q1_SELECT_WHERE": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
    """,

    "Q2_ORDER_BY_LIMIT": """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10
    """,

    "Q3_DISTINCT": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating
    """,

    "Q4_BETWEEN": """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """,

    "Q5_IN": """
        SELECT title, rating, in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC
    """,

    "Q6_JOIN": """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.book_id
        LIMIT 10
    """
}


def create_database(cleaned_df):
    """Create normalized SQLite database."""

    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)

    # Remove old database so every run starts clean
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)

    conn = sqlite3.connect(DB_NAME)

    conn.execute("PRAGMA foreign_keys = ON")

    cursor = conn.cursor()

    # Categories table
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock BOOLEAN NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # Insert categories
    categories_df = (
        cleaned_df[["category"]]
        .drop_duplicates()
        .rename(
            columns={"category": "category_name"}
        )
    )

    categories_df.to_sql(
        "categories",
        conn,
        if_exists="append",
        index=False
    )

    # Get category IDs
    category_lookup = pd.read_sql(
        """
        SELECT category_id, category_name
        FROM categories
        """,
        conn
    )

    # Add category ID to books
    books_df = cleaned_df.merge(
        category_lookup,
        left_on="category",
        right_on="category_name",
        how="left"
    )

    books_df = books_df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_id"
        ]
    ]

    books_df.to_sql(
        "books",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()

    return conn


def run_queries(conn):
    """Run required SQL queries and save outputs."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    query_outputs = {}

    output_file = os.path.join(
        OUTPUT_DIR,
        "sql_queries_and_outputs.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for query_name, query in QUERIES.items():

            result = pd.read_sql_query(
                query,
                conn
            )

            query_outputs[query_name] = result

            f.write("=" * 70 + "\n")
            f.write(query_name + "\n")
            f.write("=" * 70 + "\n")

            f.write("SQL Query:\n")
            f.write(query.strip() + "\n\n")

            f.write("Output:\n")
            f.write(
                result.to_string(index=False)
            )
            f.write("\n\n")

    return query_outputs


def compare_sql_and_pandas_join(
    conn,
    cleaned_df
):
    """Compare SQL JOIN with pandas.merge()."""

    # SQL JOIN
    sql_join = """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.book_id
    """

    sql_join_df = pd.read_sql(
        sql_join,
        conn
    )

    # Create in-memory DataFrames
    books_memory_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        conn
    )

    categories_memory_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        conn
    )

    # Pandas merge
    pandas_join_df = pd.merge(
        books_memory_df,
        categories_memory_df,
        on="category_id",
        how="inner"
    )

    pandas_join_df = pandas_join_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ].sort_values(
        "book_id"
    ).reset_index(drop=True)

    sql_join_df = sql_join_df.reset_index(
        drop=True
    )

    matches = sql_join_df.equals(
        pandas_join_df
    )

    assert matches, (
        "SQL JOIN and pandas merge do not match"
    )

    # Save comparison
    comparison_df = pd.DataFrame({
        "SQL_JOIN": sql_join_df.head(10)
        .astype(str)
        .agg(" | ".join, axis=1),

        "PANDAS_MERGE": pandas_join_df.head(10)
        .astype(str)
        .agg(" | ".join, axis=1)
    })

    comparison_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sql_vs_pandas_join.csv"
        ),
        index=False
    )

    return matches


def main():
    """Run the complete data pipeline."""

    print("===== ZEPTO DATA PIPELINE =====")

    # 1. Scrape
    print("\n1. Scraping books...")
    raw_df = scrape_books(num_pages=5)

    print(
        "Books scraped:",
        len(raw_df)
    )

    raw_df.to_csv(
        RAW_FILE,
        index=False
    )

    # 2. Clean
    print("\n2. Cleaning data...")
    cleaned_df = clean_books(raw_df)

    print(
        "Cleaned rows:",
        len(cleaned_df)
    )

    cleaned_df.to_csv(
        CLEANED_FILE,
        index=False
    )

    # 3. Validate
    print("\n3. Validating data...")
    validate_books(cleaned_df)

    print("Validation: PASS")

    # 4. Database
    print("\n4. Creating SQLite database...")
    conn = create_database(cleaned_df)

    print(
        "Database created:",
        DB_NAME
    )

    # 5. SQL queries
    print("\n5. Running SQL queries...")
    query_outputs = run_queries(conn)

    print(
        "Queries executed:",
        len(query_outputs)
    )

    # 6. SQL vs pandas JOIN
    print("\n6. Comparing SQL JOIN with pandas merge...")
    matches = compare_sql_and_pandas_join(
        conn,
        cleaned_df
    )

    print(
        "SQL JOIN = pandas merge:",
        matches
    )

    conn.close()

    print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
