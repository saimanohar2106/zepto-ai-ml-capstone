
import pandas as pd


GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def clean_books(raw_df):
    """Clean scraped book data and convert GBP prices to INR."""

    df = raw_df.copy()

    # Clean GBP price
    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Convert star rating to integer
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # Convert availability to Boolean
    df["in_stock"] = (
        df["availability"]
        .astype(str)
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

    # Median imputation for numeric fields
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(
            df["price_gbp"].median()
        )

    if df["rating"].isna().any():
        median_rating = round(df["rating"].median())

        df["rating"] = df["rating"].fillna(
            median_rating
        )

    df["rating"] = df["rating"].astype(int)

    # Fixed GBP → INR conversion
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # Remove rows missing essential fields
    df = df.dropna(
        subset=["title", "category"]
    )

    # Keep required columns
    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    return df


if __name__ == "__main__":

    raw_df = pd.read_csv(
        "data_pipeline/data/raw_books.csv"
    )

    cleaned_df = clean_books(raw_df)

    cleaned_df.to_csv(
        "data_pipeline/data/cleaned_books.csv",
        index=False
    )

    print(
        "Cleaned rows:",
        len(cleaned_df)
    )

    print(
        "GBP to INR rate:",
        GBP_TO_INR
    )
