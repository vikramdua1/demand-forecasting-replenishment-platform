from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import pandas as pd

from pipeline.validate import validate_or_raise, save_validation_report


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "retail_store_inventory.csv"
STAGING_DIR = BASE_DIR / "data" / "staging"
CURATED_DIR = BASE_DIR / "data" / "curated"
LOGS_DIR = BASE_DIR / "logs"

STAGING_PATH = STAGING_DIR / "clean_daily_inventory.csv"
WEEKLY_OUTPUT_PATH = CURATED_DIR / "weekly_modeling_table.csv"
VALIDATION_REPORT_PATH = LOGS_DIR / "raw_validation_report.json"
PIPELINE_RUN_LOG_PATH = LOGS_DIR / "ingest_run_log.json"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def clean_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    critical_columns = ["date", "store_id", "product_id", "units_sold"]
    df = df.dropna(subset=critical_columns)

    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    return df


def build_weekly_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["week_start_date"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    df["week_start_date"] = df["week_start_date"].dt.normalize()

    weekly_df = (
        df.groupby(["store_id", "product_id", "week_start_date"], as_index=False)
        .agg(
            weekly_units_sold=("units_sold", "sum"),
            weekly_units_ordered=("units_ordered", "sum"),
            avg_inventory_level=("inventory_level", "mean"),
            avg_price=("price", "mean"),
            avg_discount=("discount", "mean"),
            avg_competitor_pricing=("competitor_pricing", "mean"),
            avg_demand_forecast=("demand_forecast", "mean"),
            dominant_seasonality=("seasonality", lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA),
            dominant_weather_condition=("weather_condition", lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA),
            holiday_promotion_flag=("holiday_promotion", lambda x: int((x.astype(str).str.lower().isin(["yes", "true", "1"])).any())),
        )
    )

    weekly_df = weekly_df.sort_values(["store_id", "product_id", "week_start_date"]).reset_index(drop=True)

    return weekly_df


def write_run_log(
    raw_rows: int,
    staging_rows: int,
    curated_rows: int,
    validation_report: dict,
) -> None:
    run_log = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "raw_input_path": str(RAW_PATH),
        "staging_output_path": str(STAGING_PATH),
        "curated_output_path": str(WEEKLY_OUTPUT_PATH),
        "raw_row_count": int(raw_rows),
        "staging_row_count": int(staging_rows),
        "curated_row_count": int(curated_rows),
        "validation_status": validation_report["status"],
        "validation_warnings": validation_report["warnings"],
        "summary": validation_report["summary"],
    }

    PIPELINE_RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_RUN_LOG_PATH, "w") as f:
        json.dump(run_log, f, indent=4)


def main() -> None:
    print("Loading raw retail inventory data...")

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    raw_rows = len(df)

    print("Standardizing columns...")
    df = standardize_columns(df)

    print("Coercing data types...")
    df = coerce_types(df)

    print("Validating raw data...")
    validation_report = validate_or_raise(df)
    save_validation_report(validation_report, VALIDATION_REPORT_PATH)

    print("Cleaning daily data...")
    clean_df = clean_daily_data(df)
    staging_rows = len(clean_df)

    print(f"Saving staged daily data to {STAGING_PATH} ...")
    clean_df.to_csv(STAGING_PATH, index=False)

    print("Building weekly modeling table...")
    weekly_df = build_weekly_modeling_table(clean_df)
    curated_rows = len(weekly_df)

    print(f"Saving curated weekly modeling table to {WEEKLY_OUTPUT_PATH} ...")
    weekly_df.to_csv(WEEKLY_OUTPUT_PATH, index=False)

    print("Writing ingest run log...")
    write_run_log(
        raw_rows=raw_rows,
        staging_rows=staging_rows,
        curated_rows=curated_rows,
        validation_report=validation_report,
    )

    print("Done.")
    print(f"Raw rows: {raw_rows}")
    print(f"Staging rows: {staging_rows}")
    print(f"Curated weekly rows: {curated_rows}")

    print("\nSample weekly rows:")
    print(weekly_df.head())


if __name__ == "__main__":
    main()