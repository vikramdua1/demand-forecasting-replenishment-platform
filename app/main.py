from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException


DATA_PATH = Path("data/processed/scored_forecasts.csv")

app = FastAPI(
    title="Demand Forecasting & Replenishment API",
    description="Serves next-week demand forecasts and replenishment recommendations.",
    version="1.0.0",
)


def load_scored_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Scored forecast file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["week_start_date", "forecast_week_start_date"])
    return df


@app.get("/")
def root():
    return {"message": "Demand Forecasting & Replenishment API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecasts")
def get_all_forecasts():
    df = load_scored_data()
    return df.to_dict(orient="records")


@app.get("/forecast/{store_id}/{product_id}")
def get_forecast(store_id: str, product_id: str):
    df = load_scored_data()

    result = df[
        (df["store_id"] == store_id) &
        (df["product_id"] == product_id)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for store_id={store_id}, product_id={product_id}",
        )

    return result.iloc[0].to_dict()


@app.get("/replenishment/{store_id}/{product_id}")
def get_replenishment(store_id: str, product_id: str):
    df = load_scored_data()

    result = df[
        (df["store_id"] == store_id) &
        (df["product_id"] == product_id)
    ]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No replenishment record found for store_id={store_id}, product_id={product_id}",
        )

    row = result.iloc[0]

    return {
        "store_id": row["store_id"],
        "product_id": row["product_id"],
        "forecast_week_start_date": row["forecast_week_start_date"],
        "predicted_next_week_units_sold": row["predicted_next_week_units_sold"],
        "reorder_point": row["reorder_point"],
        "recommended_order_qty": row["recommended_order_qty"],
        "stockout_risk": row["stockout_risk"],
    }
    