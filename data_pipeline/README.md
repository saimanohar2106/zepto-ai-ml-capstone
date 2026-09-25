
# Module 1 — Data Pipeline

## Overview

This module builds an end-to-end data pipeline for scraping, cleaning,
validating, storing, and querying book product data.

The pipeline uses live data from:

https://books.toscrape.com/

The scraped data is stored in a normalized SQLite database and is also
saved as CSV and text outputs for analysis and verification.

---

## Scope

For this project, the accepted scraping scope is:

- First 5 paginated pages of the All Products catalogue
- 20 books per page
- Total expected records: 100 books
- Scraping is fully automated
- No manual copy/paste of product data

The source website provides book-level product information including
title, price, star rating, and availability.

---

## Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- unittest

SQLite is included with Python and does not require a separate installation.

---

## Project Structure

```text
data_pipeline/
├── src/
│   ├── scraper.py
│   ├── cleaner.py
│   ├── validator.py
│   └── pipeline.py
│
├── data/
│   ├── raw_books.csv
│   ├── cleaned_books.csv
│   └── zepto_books.db
│
├── outputs/
│   ├── sql_queries_and_outputs.txt
│   ├── sql_vs_pandas_join.csv
│   └── validation_summary.csv
│
├── tests/
│   └── test_pipeline.py
│
└── README.md
