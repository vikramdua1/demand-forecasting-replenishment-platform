from pathlib import Path
import pickle

import pandas as pd


INPUT_PATH = Path("data/processed/model_feature_table.csv")
MODEL_PATH = Path("models/random_forest_model.pkl")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "scored_forecasts.csv"


FEATURE_COLUMNS = [
    "store_id",
    "product_id",
    "dominant_seasonality",
    "holiday_promotion_flag",
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


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    return pd.read_csv(path, parse_dates=["week_start_date"])


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the latest available row for each store-product pair.
    These are the rows used to predict the next week.
    """
    latest = (
        df.sort_values(["store_id", "product_id", "week_start_date"])
          .groupby(["store_id", "product_id"], as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )
    return latest


def add_next_forecast_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["forecast_week_start_date"] = df["week_start_date"] + pd.Timedelta(days=7)
    return df


def generate_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURE_COLUMNS]
    df["predicted_next_week_units_sold"] = model.predict(X).round(2)
    return df


def add_replenishment_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple v1 replenishment logic.

    Assumptions:
    - current inventory proxy = this week's sales (lag_1) is not enough
      so use a simple coverage rule based on forecast + safety stock
    - reorder point = forecast + 0.5 * rolling std
    - recommended qty = max(0, reorder point - lag_1 proxy)
    """
    df = df.copy()

    # Using lag_1 as a crude proxy for recent observed demand level
    df["reorder_point"] = (
        df["predicted_next_week_units_sold"] + 0.5 * df["rolling_std_4"]
    ).round(2)

    df["recommended_order_qty"] = (
        df["reorder_point"] - df["lag_1"]
    ).clip(lower=0).round(2)

    def classify_risk(row) -> str:
        if row["lag_1"] < 0.8 * row["predicted_next_week_units_sold"]:
            return "high"
        if row["lag_1"] < row["predicted_next_week_units_sold"]:
            return "medium"
        return "low"

    df["stockout_risk"] = df.apply(classify_risk, axis=1)

    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "store_id",
            "product_id",
            "week_start_date",
            "forecast_week_start_date",
            "weekly_units_sold",
            "lag_1",
            "rolling_mean_4",
            "rolling_std_4",
            "predicted_next_week_units_sold",
            "reorder_point",
            "recommended_order_qty",
            "stockout_risk",
            "dominant_seasonality",
            "holiday_promotion_flag",
        ]
    ].copy()


def save_output(df: pd.DataFrame, path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    print("Loading feature table...")
    df = load_feature_table(INPUT_PATH)

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Selecting latest store-product rows...")
    latest_df = get_latest_rows(df)

    print("Adding next forecast week...")
    latest_df = add_next_forecast_week(latest_df)

    print("Generating predictions...")
    latest_df = generate_predictions(model, latest_df)

    print("Applying replenishment logic...")
    latest_df = add_replenishment_logic(latest_df)

    print("Selecting final output columns...")
    output_df = select_output_columns(latest_df)

    print(f"Saving scored forecasts to {OUTPUT_PATH} ...")
    save_output(output_df, OUTPUT_PATH)

    print("Done.")
    print(f"Output shape: {output_df.shape}")
    print("\nSample rows:")
    print(output_df.head())


if __name__ == "__main__":
    main()
    