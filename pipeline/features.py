from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "curated" / "weekly_modeling_table.csv"
OUTPUT_DIR = BASE_DIR / "data" / "curated"
OUTPUT_PATH = OUTPUT_DIR / "model_feature_table.csv"


def load_weekly_data(path: Path) -> pd.DataFrame:
    """Load the weekly modeling table."""
    if not path.exists():
        raise FileNotFoundError(f"Weekly modeling table not found: {path}")

    df = pd.read_csv(path, parse_dates=["week_start_date"])
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort data by store, product, and week."""
    df = df.sort_values(["store_id", "product_id", "week_start_date"]).reset_index(drop=True)
    return df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag and rolling features within each store-product series."""
    df = df.copy()

    group_cols = ["store_id", "product_id"]
    target_col = "weekly_units_sold"

    grouped = df.groupby(group_cols)[target_col]

    df["lag_1"] = grouped.shift(1)
    df["lag_2"] = grouped.shift(2)
    df["lag_4"] = grouped.shift(4)

    df["rolling_mean_4"] = grouped.shift(1).rolling(window=4, min_periods=1).mean()
    df["rolling_std_4"] = grouped.shift(1).rolling(window=4, min_periods=1).std()

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-derived features from the weekly date."""
    df = df.copy()

    iso_calendar = df["week_start_date"].dt.isocalendar()
    df["year"] = df["week_start_date"].dt.year
    df["month"] = df["week_start_date"].dt.month
    df["quarter"] = df["week_start_date"].dt.quarter
    df["week_of_year"] = iso_calendar.week.astype(int)

    return df


def add_forecast_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create next-week forecasting target within each store-product series.

    Each row will represent information known at week t, with target equal to
    sales at week t+1.
    """
    df = df.copy()

    df["target_next_week"] = (
        df.groupby(["store_id", "product_id"])["weekly_units_sold"]
        .shift(-1)
    )

    return df


def clean_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with insufficient history for lag-based modeling and rows where
    next-week target is unavailable.
    """
    df = df.copy()

    df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "target_next_week"])
    df["rolling_std_4"] = df["rolling_std_4"].fillna(0)

    return df.reset_index(drop=True)


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select columns for the leakage-safe forecasting table.

    We intentionally exclude potentially leaky same-week operational variables
    such as weekly_units_ordered, avg_inventory_level, avg_price, avg_discount,
    avg_competitor_pricing, and avg_demand_forecast.
    """
    selected_columns = [
        "week_start_date",
        "store_id",
        "product_id",
        "weekly_units_sold",
        "target_next_week",
        "holiday_promotion_flag",
        "dominant_weather_condition",
        "dominant_seasonality",
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "year",
        "month",
        "quarter",
        "week_of_year",
    ]

    return df[selected_columns].copy()


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save the model feature table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    print("Loading weekly modeling table...")
    df = load_weekly_data(INPUT_PATH)

    print("Sorting data...")
    df = sort_data(df)

    print("Adding lag and rolling features...")
    df = add_time_series_features(df)

    print("Adding calendar features...")
    df = add_calendar_features(df)

    print("Creating next-week forecasting target...")
    df = add_forecast_target(df)

    print("Cleaning feature table...")
    df = clean_feature_table(df)

    print("Selecting modeling columns...")
    df = select_model_columns(df)

    print(f"Saving feature table to {OUTPUT_PATH} ...")
    save_output(df, OUTPUT_PATH)

    print("Done.")
    print(f"Output shape: {df.shape}")
    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
