from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "category",
    "region",
    "inventory_level",
    "units_sold",
    "units_ordered",
    "demand_forecast",
    "price",
    "discount",
    "weather_condition",
    "holiday_promotion",
    "competitor_pricing",
    "seasonality",
]

NUMERIC_COLUMNS = [
    "inventory_level",
    "units_sold",
    "units_ordered",
    "demand_forecast",
    "price",
    "discount",
    "competitor_pricing",
]


def build_validation_report(df: pd.DataFrame) -> dict:
    """
    Build a validation report for the raw/staged retail dataset.
    The function does not mutate the input dataframe.
    """
    report: dict = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "status": "passed",
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    # Required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        report["status"] = "failed"
        report["errors"].append(
            f"Missing required columns: {missing_columns}"
        )
        return report

    # Date parsing check
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates > 0:
        report["status"] = "failed"
        report["errors"].append(
            f"Invalid date values found: {invalid_dates}"
        )

    # Duplicate business key check
    duplicate_count = int(
        df.duplicated(subset=["store_id", "product_id", "date"]).sum()
    )
    if duplicate_count > 0:
        report["status"] = "failed"
        report["errors"].append(
            f"Duplicate (store_id, product_id, date) rows found: {duplicate_count}"
        )

    # Numeric coercion + negative checks
    for col in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        null_after_coercion = int(coerced.isna().sum())
        if null_after_coercion > 0:
            report["status"] = "failed"
            report["errors"].append(
                f"Non-numeric or null values found in {col}: {null_after_coercion}"
            )

    non_negative_cols = ["inventory_level", "units_sold", "units_ordered", "price", "discount"]
    for col in non_negative_cols:
        negative_count = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
        if negative_count > 0:
            report["status"] = "failed"
            report["errors"].append(
                f"Negative values found in {col}: {negative_count}"
            )

    # Category instability check
    product_category_nunique = df.groupby("product_id")["category"].nunique()
    unstable_products = int((product_category_nunique > 1).sum())
    if unstable_products > 0:
        report["warnings"].append(
            f"{unstable_products} product_ids map to multiple categories."
        )

    # Region instability check
    store_region_nunique = df.groupby("store_id")["region"].nunique()
    unstable_stores = int((store_region_nunique > 1).sum())
    if unstable_stores > 0:
        report["warnings"].append(
            f"{unstable_stores} store_ids map to multiple regions."
        )

    # Null summary
    null_counts = df.isna().sum()
    null_summary = {k: int(v) for k, v in null_counts[null_counts > 0].to_dict().items()}
    if null_summary:
        report["warnings"].append(
            f"Null values present in columns: {null_summary}"
        )

    # Summary fields
    report["summary"] = {
        "date_min": str(parsed_dates.min()) if invalid_dates < len(df) else None,
        "date_max": str(parsed_dates.max()) if invalid_dates < len(df) else None,
        "unique_stores": int(df["store_id"].nunique()),
        "unique_products": int(df["product_id"].nunique()),
        "duplicate_key_rows": duplicate_count,
        "invalid_dates": invalid_dates,
        "unstable_products": unstable_products,
        "unstable_stores": unstable_stores,
    }

    return report


def validate_or_raise(df: pd.DataFrame) -> dict:
    """
    Validate the dataset and raise an error if critical checks fail.
    Returns the validation report if validation succeeds.
    """
    report = build_validation_report(df)

    if report["status"] == "failed":
        error_text = "\n".join(report["errors"])
        raise ValueError(f"Validation failed:\n{error_text}")

    return report


def save_validation_report(report: dict, output_path: Path) -> None:
    """Save validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)