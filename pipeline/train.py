from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_PATH = BASE_DIR / "data" / "curated" / "model_feature_table.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"


TARGET_COL = "target_next_week"
DATE_COL = "week_start_date"

CATEGORICAL_COLS = [
    "store_id",
    "product_id",
    "dominant_seasonality",
]

NUMERICAL_COLS = [
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


def load_data(path: Path) -> pd.DataFrame:
    """Load the model feature table."""
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time so later periods form the test set."""
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    unique_dates = df[DATE_COL].sort_values().unique()
    split_idx = int(len(unique_dates) * (1 - test_fraction))
    split_date = unique_dates[split_idx]

    train_df = df[df[DATE_COL] < split_date].copy()
    test_df = df[df[DATE_COL] >= split_date].copy()

    return train_df, test_df


def compute_metrics(y_true: pd.Series, y_pred) -> dict:
    """Compute evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def evaluate_baselines(test_df: pd.DataFrame) -> dict:
    """
    Evaluate simple next-week forecasting baselines.

    Baseline 1:
    Predict next week using this week's actual sales.

    Baseline 2:
    Predict next week using the rolling mean of recent prior weeks.
    """
    y_true = test_df[TARGET_COL]

    baseline_results = {}

    pred_current_week = test_df["weekly_units_sold"]
    baseline_results["baseline_current_week_equals_next_week"] = compute_metrics(y_true, pred_current_week)

    pred_rolling_mean_4 = test_df["rolling_mean_4"]
    baseline_results["baseline_rolling_mean_4"] = compute_metrics(y_true, pred_rolling_mean_4)

    return baseline_results


def evaluate_holt_winters(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Fit Holt-Winters separately for each store-product series and evaluate on the test period.

    Uses weekly_units_sold from the training set only, then forecasts forward for the
    number of test rows in each series.
    """
    predictions = []
    actuals = []

    grouped_test = test_df.groupby(["store_id", "product_id"])

    successful_series = 0
    failed_series = 0

    for (store_id, product_id), test_group in grouped_test:
        train_group = train_df[
            (train_df["store_id"] == store_id) &
            (train_df["product_id"] == product_id)
        ].sort_values(DATE_COL)

        test_group = test_group.sort_values(DATE_COL)

        train_series = train_group["weekly_units_sold"]

        if len(train_series) < 8:
            failed_series += 1
            continue

        try:
            model = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )

            fitted_model = model.fit(optimized=True)
            forecast_horizon = len(test_group)
            forecast_values = fitted_model.forecast(forecast_horizon)

            predictions.extend(forecast_values.tolist())
            actuals.extend(test_group[TARGET_COL].tolist())
            successful_series += 1

        except Exception:
            failed_series += 1
            continue

    if len(actuals) == 0:
        raise ValueError("Holt-Winters evaluation produced no valid forecasts.")

    metrics = compute_metrics(pd.Series(actuals), pd.Series(predictions))
    metrics["successful_series"] = int(successful_series)
    metrics["failed_series"] = int(failed_series)

    return metrics


def build_model_pipeline() -> Pipeline:
    """Build preprocessing + Random Forest pipeline."""
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numerical_transformer, NUMERICAL_COLS),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """Train the forecasting model."""
    X_train = train_df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y_train = train_df[TARGET_COL]

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model: Pipeline, test_df: pd.DataFrame) -> dict:
    """Evaluate the trained Random Forest model."""
    X_test = test_df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y_test = test_df[TARGET_COL]

    predictions = model.predict(X_test)
    return compute_metrics(y_test, predictions)


def save_model(model: Pipeline, path: Path) -> None:
    """Save the trained model artifact."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_metrics(metrics: dict, path: Path) -> None:
    """Save training and evaluation metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def main() -> None:
    print("Loading feature table...")
    df = load_data(FEATURE_PATH)

    print("Creating time-based split...")
    train_df, test_df = time_based_split(df)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train date range: {train_df[DATE_COL].min().date()} to {train_df[DATE_COL].max().date()}")
    print(f"Test date range: {test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()}")

    print("Evaluating baseline models...")
    baseline_metrics = evaluate_baselines(test_df)

    print("Evaluating Holt-Winters benchmark...")
    holt_winters_metrics = evaluate_holt_winters(train_df, test_df)

    print("Training Random Forest model...")
    model = train_model(train_df)

    print("Evaluating Random Forest model...")
    model_metrics = evaluate_model(model, test_df)

    all_metrics = {
        "target_definition": "predict next week's units sold using information available at the current week",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS,
        "baselines": baseline_metrics,
        "holt_winters": holt_winters_metrics,
        "random_forest": model_metrics,
    }

    print("Saving model artifact...")
    save_model(model, MODEL_PATH)

    print("Saving metrics...")
    save_metrics(all_metrics, METRICS_PATH)

    print("\nTraining complete.\n")
    print("Baseline metrics:")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nHolt-Winters metrics:")
    print(json.dumps(holt_winters_metrics, indent=4))

    print("\nRandom Forest metrics:")
    print(json.dumps(model_metrics, indent=4))

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()