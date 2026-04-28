from pathlib import Path
import pandas as pd


RAW_PATH = Path("/Users/admin/Downloads/demand-forecasting-platform/data/raw/retail_store_inventory.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "weekly_modeling_table.csv"


COLUMN_RENAME_MAP = {
    "Date": "date",
    "Store ID": "store_id",
    "Product ID": "product_id",
    "Category": "category",
    "Region": "region",
    "Inventory Level": "inventory_level",
    "Units Sold": "units_sold",
    "Units Ordered": "units_ordered",
    "Demand Forecast": "demand_forecast",
    "Price": "price",
    "Discount": "discount",
    "Weather Condition": "weather_condition",
    "Holiday/Promotion": "holiday_promotion",
    "Competitor Pricing": "competitor_pricing",
    "Seasonality": "seasonality",
}


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw retail inventory dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case names used across the project."""
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def cast_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, cast numeric columns, and run basic validation checks."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "inventory_level",
        "units_sold",
        "units_ordered",
        "demand_forecast",
        "price",
        "discount",
        "competitor_pricing",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required_columns = [
        "date",
        "store_id",
        "product_id",
        "category",
        "region",
        "inventory_level",
        "units_sold",
        "units_ordered",
        "price",
        "discount",
        "competitor_pricing",
        "weather_condition",
        "holiday_promotion",
        "seasonality",
    ]

    missing_required = df[required_columns].isnull().sum()
    if missing_required.any():
        raise ValueError(
            "Missing values found in required columns:\n"
            f"{missing_required[missing_required > 0]}"
        )

    if (df["units_sold"] < 0).any():
        raise ValueError("Negative values found in units_sold.")

    if (df["inventory_level"] < 0).any():
        raise ValueError("Negative values found in inventory_level.")

    return df


def add_week_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create a week start date column for weekly aggregation."""
    df = df.copy()
    df["week_start_date"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    return df


def build_weekly_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily dataset to store-product-week level."""
    weekly = (
        df.groupby(["week_start_date", "store_id", "product_id"], as_index=False)
        .agg(
            category=("category", lambda x: x.mode().iloc[0]),
            region=("region", lambda x: x.mode().iloc[0]),
            weekly_units_sold=("units_sold", "sum"),
            weekly_units_ordered=("units_ordered", "sum"),
            avg_inventory_level=("inventory_level", "mean"),
            avg_price=("price", "mean"),
            avg_discount=("discount", "mean"),
            avg_competitor_pricing=("competitor_pricing", "mean"),
            avg_demand_forecast=("demand_forecast", "mean"),
            holiday_promotion_flag=("holiday_promotion", lambda x: int((x == 1).any())),
            dominant_weather_condition=("weather_condition", lambda x: x.mode().iloc[0]),
            dominant_seasonality=("seasonality", lambda x: x.mode().iloc[0]),
            days_in_week=("date", "nunique"),
        )
        .sort_values(["store_id", "product_id", "week_start_date"])
        .reset_index(drop=True)
    )

    return weekly


def remove_incomplete_edge_weeks(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove incomplete edge weeks.
    Since weekly resampling may create partial first/last weeks,
    keep only records with full 7-day coverage.
    """
    cleaned = weekly_df[weekly_df["days_in_week"] == 7].copy()
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save the processed weekly modeling table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    print("Loading raw data...")
    df = load_raw_data(RAW_PATH)

    print("Standardizing columns...")
    df = standardize_columns(df)

    print("Casting and validating data...")
    df = cast_and_validate(df)

    print("Adding weekly time keys...")
    df = add_week_columns(df)

    print("Building weekly modeling table...")
    weekly_df = build_weekly_modeling_table(df)

    print("Removing incomplete edge weeks...")
    weekly_df = remove_incomplete_edge_weeks(weekly_df)

    print(f"Saving weekly modeling table to {OUTPUT_PATH} ...")
    save_output(weekly_df, OUTPUT_PATH)

    print("Done.")
    print(f"Output shape: {weekly_df.shape}")
    print("\nSample rows:")
    print(weekly_df.head())


if __name__ == "__main__":
    main()