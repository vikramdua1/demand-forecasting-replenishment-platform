import os
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero-card {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1f2937 100%);
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.16);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 0.96rem;
        color: rgba(255,255,255,0.82);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
    }

    .insight-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1rem 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }

    .small-label {
        font-size: 0.82rem;
        opacity: 0.8;
        margin-bottom: 0.2rem;
    }

    .risk-high {
        color: #ef4444;
        font-weight: 700;
    }

    .risk-medium {
        color: #f59e0b;
        font-weight: 700;
    }

    .risk-low {
        color: #10b981;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_forecasts():
    response = requests.get(f"{API_BASE_URL}/forecasts", timeout=20)
    response.raise_for_status()
    df = pd.DataFrame(response.json())

    if "week_start_date" in df.columns:
        df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    if "forecast_week_start_date" in df.columns:
        df["forecast_week_start_date"] = pd.to_datetime(df["forecast_week_start_date"])

    return df


@st.cache_data
def load_single_forecast(store_id: str, product_id: str):
    response = requests.get(
        f"{API_BASE_URL}/forecast/{store_id}/{product_id}",
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def format_risk_html(risk: str) -> str:
    risk = str(risk).lower()
    if risk == "high":
        return '<span class="risk-high">High</span>'
    if risk == "medium":
        return '<span class="risk-medium">Medium</span>'
    return '<span class="risk-low">Low</span>'


def build_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_metric_row(forecast: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Next Week Units", f"{forecast['predicted_next_week_units_sold']:.2f}")
    c2.metric("Reorder Point", f"{forecast['reorder_point']:.2f}")
    c3.metric("Recommended Order Qty", f"{forecast['recommended_order_qty']:.2f}")
    c4.markdown(
        f"""
        <div class="insight-card">
            <div class="small-label">Stockout Risk</div>
            <div style="font-size:1.45rem;">{format_risk_html(forecast["stockout_risk"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_panel(forecast: dict):
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="section-title">Forecast Detail</div>', unsafe_allow_html=True)
        detail_df = pd.DataFrame(
            {
                "Field": [
                    "Store ID",
                    "Product ID",
                    "Current Week Start",
                    "Forecast Week Start",
                    "Current Week Units Sold",
                    "Lag 1",
                    "Rolling Mean (4)",
                    "Rolling Std (4)",
                    "Predicted Next Week Units Sold",
                    "Reorder Point",
                    "Recommended Order Qty",
                    "Dominant Seasonality",
                    "Holiday/Promotion Flag",
                ],
                "Value": [
                    forecast["store_id"],
                    forecast["product_id"],
                    forecast["week_start_date"],
                    forecast["forecast_week_start_date"],
                    forecast["weekly_units_sold"],
                    forecast["lag_1"],
                    round(forecast["rolling_mean_4"], 2),
                    round(forecast["rolling_std_4"], 2),
                    round(forecast["predicted_next_week_units_sold"], 2),
                    round(forecast["reorder_point"], 2),
                    round(forecast["recommended_order_qty"], 2),
                    forecast["dominant_seasonality"],
                    forecast["holiday_promotion_flag"],
                ],
            }
        )
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="section-title">Decision Support Summary</div>', unsafe_allow_html=True)

        demand_gap = forecast["predicted_next_week_units_sold"] - forecast["lag_1"]
        summary_text = "Recent sales are comfortably covering the forecast."
        if demand_gap > 0:
            summary_text = "Expected demand is above recent observed demand, which supports a replenishment action."

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="small-label">Recommendation</div>
                <div style="font-size:1.05rem; font-weight:600; margin-bottom:0.5rem;">
                    Order {forecast["recommended_order_qty"]:.2f} units
                </div>
                <div style="font-size:0.92rem; opacity:0.88;">
                    {summary_text}
                </div>
            </div>

            <div class="insight-card">
                <div class="small-label">Demand Gap vs Recent Week</div>
                <div style="font-size:1.25rem; font-weight:700;">
                    {demand_gap:.2f}
                </div>
            </div>

            <div class="insight-card">
                <div class="small-label">Seasonality Context</div>
                <div style="font-size:1.05rem; font-weight:600;">
                    {forecast["dominant_seasonality"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_trend_chart(forecast: dict):
    st.markdown('<div class="section-title">Recent Demand Context</div>', unsafe_allow_html=True)

    chart_df = pd.DataFrame(
        {
            "Metric": ["Lag 4", "Lag 2", "Lag 1", "Rolling Mean (4)", "Forecast"],
            "Units": [
                forecast["lag_4"],
                forecast["lag_2"],
                forecast["lag_1"],
                forecast["rolling_mean_4"],
                forecast["predicted_next_week_units_sold"],
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chart_df["Metric"], chart_df["Units"], marker="o")
    ax.set_ylabel("Units")
    ax.set_xlabel("")
    ax.set_title("Recent Demand and Forecast")
    ax.grid(alpha=0.3)
    st.pyplot(fig)


def executive_overview_tab(df: pd.DataFrame):
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)

    high_risk_df = df[df["stockout_risk"] == "high"].copy()
    top_replenishment_df = df.sort_values("recommended_order_qty", ascending=False).head(10).copy()

    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])

    with c1:
        st.markdown("#### Top High-Risk Items")
        high_risk_display = high_risk_df[
            [
                "store_id",
                "product_id",
                "predicted_next_week_units_sold",
                "recommended_order_qty",
                "stockout_risk",
            ]
        ].head(10)
        st.dataframe(high_risk_display, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Top Replenishment Priorities")
        replenish_display = top_replenishment_df[
            [
                "store_id",
                "product_id",
                "predicted_next_week_units_sold",
                "reorder_point",
                "recommended_order_qty",
            ]
        ]
        st.dataframe(replenish_display, use_container_width=True, hide_index=True)

    with c3:
        st.markdown("#### Portfolio Summary")
        risk_counts = (
            df["stockout_risk"]
            .value_counts()
            .rename_axis("risk")
            .reset_index(name="count")
        )
        st.dataframe(risk_counts, use_container_width=True, hide_index=True)

        summary_by_store = (
            df.groupby("store_id", as_index=False)
              .agg(
                  total_recommended_order_qty=("recommended_order_qty", "sum"),
                  avg_forecast=("predicted_next_week_units_sold", "mean"),
                  high_risk_items=("stockout_risk", lambda x: int((x == "high").sum())),
              )
              .sort_values("total_recommended_order_qty", ascending=False)
        )

        st.markdown("#### Store-Level Priority Summary")
        st.dataframe(summary_by_store, use_container_width=True, hide_index=True)


def forecast_explorer_tab(df: pd.DataFrame):
    st.markdown('<div class="section-title">Forecast Explorer</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        selected_store = st.selectbox("Select Store", sorted(df["store_id"].unique()))

    with col2:
        filtered_products = df[df["store_id"] == selected_store]["product_id"].unique()
        selected_product = st.selectbox("Select Product", sorted(filtered_products))

    forecast = load_single_forecast(selected_store, selected_product)

    render_metric_row(forecast)
    render_trend_chart(forecast)
    render_detail_panel(forecast)


def portfolio_view_tab(df: pd.DataFrame):
    st.markdown('<div class="section-title">Portfolio View</div>', unsafe_allow_html=True)

    filter1, filter2, filter3 = st.columns(3)

    with filter1:
        store_filter = st.multiselect(
            "Filter by Store",
            options=sorted(df["store_id"].unique()),
            default=sorted(df["store_id"].unique()),
        )

    with filter2:
        risk_filter = st.multiselect(
            "Filter by Stockout Risk",
            options=sorted(df["stockout_risk"].unique()),
            default=sorted(df["stockout_risk"].unique()),
        )

    with filter3:
        sort_option = st.selectbox(
            "Sort By",
            ["recommended_order_qty", "predicted_next_week_units_sold", "reorder_point"],
        )

    filtered_df = df[
        (df["store_id"].isin(store_filter)) &
        (df["stockout_risk"].isin(risk_filter))
    ].copy()

    filtered_df = filtered_df.sort_values(sort_option, ascending=False)

    s1, s2, s3 = st.columns(3)
    s1.metric("Filtered Rows", len(filtered_df))
    s2.metric("High Risk in Selection", int((filtered_df["stockout_risk"] == "high").sum()))
    s3.metric("Selected Recommended Qty", f"{filtered_df['recommended_order_qty'].sum():.2f}")

    download_bytes = build_download_bytes(filtered_df)
    st.download_button(
        label="Download filtered portfolio CSV",
        data=download_bytes,
        file_name="filtered_portfolio_view.csv",
        mime="text/csv",
    )

    display_cols = [
        "store_id",
        "product_id",
        "forecast_week_start_date",
        "predicted_next_week_units_sold",
        "reorder_point",
        "recommended_order_qty",
        "stockout_risk",
        "dominant_seasonality",
    ]

    st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)


def main():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Demand Forecasting & Replenishment Dashboard</div>
            <div class="hero-subtitle">
                Forecast next-week demand, identify stockout risk, and review replenishment actions across store-product combinations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = load_all_forecasts()
    except Exception as e:
        st.error(f"Could not connect to API: {e}")
        st.stop()

    total_items = len(df)
    high_risk_items = int((df["stockout_risk"] == "high").sum())
    total_recommended = float(df["recommended_order_qty"].sum())
    avg_forecast = float(df["predicted_next_week_units_sold"].mean())

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Store-Product Forecasts", total_items)
    top2.metric("High Risk Items", high_risk_items)
    top3.metric("Total Recommended Order Qty", f"{total_recommended:.2f}")
    top4.metric("Avg Forecasted Demand", f"{avg_forecast:.2f}")

    tab1, tab2, tab3 = st.tabs(["Executive Overview", "Forecast Explorer", "Portfolio View"])

    with tab1:
        executive_overview_tab(df)

    with tab2:
        forecast_explorer_tab(df)

    with tab3:
        portfolio_view_tab(df)


if __name__ == "__main__":
    main()