
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd


BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"


def scrape_books(num_pages=5):
    """Scrape books from the first five catalogue pages."""

    all_books = []

    for page in range(1, num_pages + 1):
        url = BASE_URL.format(page)

        response = requests.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.select("article.product_pod")

        for book in books:
            title_tag = book.select_one("h3 a")
            price_tag = book.select_one(".price_color")
            rating_tag = book.select_one("p.star-rating")
            availability_tag = book.select_one(".availability")

            title = (
                title_tag.get("title", "").strip()
                if title_tag
                else ""
            )

            price = (
                price_tag.get_text(strip=True)
                if price_tag
                else ""
            )

            star_rating = ""

            if rating_tag:
                classes = rating_tag.get("class", [])

                for rating in ["One", "Two", "Three", "Four", "Five"]:
                    if rating in classes:
                        star_rating = rating
                        break

            availability = (
                availability_tag.get_text(" ", strip=True)
                if availability_tag
                else ""
            )

            all_books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": "Books"
            })

    return pd.DataFrame(all_books)


if __name__ == "__main__":
    df = scrape_books()

    print("Total books scraped:", len(df))

    df.to_csv(
        "data_pipeline/data/raw_books.csv",
        index=False
    )
