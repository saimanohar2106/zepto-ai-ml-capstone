
import pandas as pd


GBP_TO_INR = 105.50

REQUIRED_COLUMNS = [
    "title",
    "price_gbp",
    "price_inr",
    "rating",
    "in_stock",
    "category"
]


def validate_books(df):
    """Validate the cleaned book dataset."""

    # Check minimum row count
    assert len(df) >= 60, (
        f"Expected at least 60 rows, got {len(df)}"
    )

    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    # Check missing values
    assert (
        df[REQUIRED_COLUMNS]
        .isnull()
        .sum()
        .sum()
        == 0
    ), "Dataset contains missing values"

    # Check GBP prices
    assert (
        df["price_gbp"] >= 0
    ).all(), "Invalid GBP price"

    # Check ratings
    assert (
        df["rating"].between(1, 5)
    ).all(), "Rating must be between 1 and 5"

    # Check Boolean stock field
    assert (
        df["in_stock"].dtype == bool
    ), "in_stock must be Boolean"

    # Check INR conversion
    expected_inr = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    assert df["price_inr"].equals(
        expected_inr
    ), "Incorrect GBP to INR conversion"

    return True


if __name__ == "__main__":

    df = pd.read_csv(
        "data_pipeline/data/cleaned_books.csv"
    )

    # CSV may load Boolean values as bool,
    # but explicitly convert for validation.
    df["in_stock"] = df["in_stock"].astype(bool)

    validate_books(df)

    print("===== VALIDATION PASSED =====")
    print("Rows:", len(df))
    print("GBP to INR rate:", GBP_TO_INR)
