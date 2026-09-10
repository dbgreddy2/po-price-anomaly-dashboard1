from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


APP_TITLE = "PO Price Anomaly Monitoring"
DATA_FILE = Path(__file__).with_name("PO_PRICE_ANOMALY_DATASPHERE.csv")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce5ea;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 2px 7px rgba(18, 48, 74, 0.07);
    }
    [data-testid="stMetricLabel"] {color: #5b6972;}
    h1, h2, h3 {color: #12304a;}
    .subtitle {color: #5b6972; margin-top: -0.7rem; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} was not found. Upload it to the same GitHub folder as streamlit_app.py."
        )

    df = pd.read_csv(path, dtype={"PURCHASE_ORDER": "string", "PO_ITEM": "string"})
    df["PO_DATE"] = pd.to_datetime(df["PO_DATE"], errors="coerce")

    numeric_columns = [
        "ORDER_QTY",
        "UNIT_PRICE_LOCAL",
        "INFO_UNIT_PRICE",
        "INFO_RECORD_VALID_FLAG",
        "PRICE_COMPLIANCE_FLAG",
        "PRICE_LEAKAGE_VALUE",
        "REFERENCE_ITEM_VALUE",
        "PREVIOUS_UNIT_PRICE",
        "HIST_AVG_UNIT_PRICE",
        "HIST_PRICE_DEVIATION_PCT",
        "CROSS_SUPPLIER_6M_AVG_PRICE",
        "CROSS_SUPPLIER_6M_MIN_PRICE",
        "CROSS_SUPPLIER_6M_MAX_PRICE",
        "CROSS_SUPPLIER_6M_SUPPLIER_COUNT",
        "CROSS_SUPPLIER_6M_DEVIATION_PCT",
        "ML_ANOMALY_SCORE",
        "ML_ANOMALY_FLAG",
        "ANOMALY_THRESHOLD",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["PO_MONTH"] = df["PO_DATE"].dt.to_period("M").astype("string")
    return df


def values_for(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


try:
    source_df = load_data(DATA_FILE)
except Exception as exc:
    st.error(str(exc))
    st.stop()


st.title(APP_TITLE)
st.markdown(
    '<p class="subtitle">Interactive monitoring of PySpark KMeans price-anomaly predictions</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filters")
    selected_companies = st.multiselect("Company code", values_for(source_df, "COMPANY_CODE"))
    selected_plants = st.multiselect("Plant", values_for(source_df, "PLANT"))
    selected_orgs = st.multiselect("Purchasing organization", values_for(source_df, "PURCHASING_ORG"))
    selected_suppliers = st.multiselect("Supplier", values_for(source_df, "SUPPLIER"))
    selected_materials = st.multiselect("Material", values_for(source_df, "MATERIAL"))
    selected_risks = st.multiselect("Risk level", values_for(source_df, "ML_RISK_LEVEL"))
    selected_currencies = st.multiselect("Currency", values_for(source_df, "LOCAL_CURRENCY"))

    min_date = source_df["PO_DATE"].min().date()
    max_date = source_df["PO_DATE"].max().date()
    selected_dates = st.date_input(
        "PO date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    st.caption("Filters affect every KPI, chart, and detail record.")


filtered_df = source_df.copy()
filter_map = {
    "COMPANY_CODE": selected_companies,
    "PLANT": selected_plants,
    "PURCHASING_ORG": selected_orgs,
    "SUPPLIER": selected_suppliers,
    "MATERIAL": selected_materials,
    "ML_RISK_LEVEL": selected_risks,
    "LOCAL_CURRENCY": selected_currencies,
}
for column, selected in filter_map.items():
    if selected:
        filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected)]

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_df = filtered_df[
        filtered_df["PO_DATE"].dt.date.between(start_date, end_date)
    ]


total_items = len(filtered_df)
anomaly_count = int(filtered_df["ML_ANOMALY_FLAG"].fillna(0).sum())
anomaly_rate = (100 * anomaly_count / total_items) if total_items else 0.0
high_risk_count = int((filtered_df["ML_RISK_LEVEL"] == "HIGH").sum())
noncompliant_count = int((filtered_df["PRICE_COMPLIANCE_FLAG"] == 0).sum())

currencies_in_scope = values_for(filtered_df, "LOCAL_CURRENCY")
if len(currencies_in_scope) == 1:
    leakage_value = filtered_df["PRICE_LEAKAGE_VALUE"].fillna(0).sum()
    leakage_display = f"{leakage_value:,.2f} {currencies_in_scope[0]}"
else:
    leakage_display = "Select one currency"

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Scored PO items", f"{total_items:,}")
k2.metric("Detected anomalies", f"{anomaly_count:,}")
k3.metric("Anomaly rate", f"{anomaly_rate:.2f}%")
k4.metric("High-risk items", f"{high_risk_count:,}")
k5.metric("Price noncompliant", f"{noncompliant_count:,}")
k6.metric("Potential leakage", leakage_display)

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

anomalies_df = filtered_df[filtered_df["ML_ANOMALY_FLAG"] == 1].copy()

overview_tab, review_tab, model_tab = st.tabs(
    ["Executive overview", "Anomaly review", "Model information"]
)

with overview_tab:
    left, right = st.columns(2)

    risk_summary = (
        filtered_df.groupby("ML_RISK_LEVEL", dropna=False)
        .size()
        .reset_index(name="RECORD_COUNT")
    )
    risk_figure = px.pie(
        risk_summary,
        names="ML_RISK_LEVEL",
        values="RECORD_COUNT",
        title="Risk-level distribution",
        hole=0.48,
        color="ML_RISK_LEVEL",
        color_discrete_map={"HIGH": "#c0392b", "MEDIUM": "#e8a23a", "LOW": "#2a9d8f"},
    )
    risk_figure.update_layout(legend_title_text="Risk", margin=dict(t=55, b=10, l=10, r=10))
    left.plotly_chart(risk_figure, use_container_width=True)

    monthly_summary = (
        anomalies_df.groupby("PO_MONTH").size().reset_index(name="ANOMALY_COUNT")
    )
    monthly_figure = px.line(
        monthly_summary,
        x="PO_MONTH",
        y="ANOMALY_COUNT",
        markers=True,
        title="Monthly detected-anomaly trend",
    )
    monthly_figure.update_traces(line_color="#147d92", line_width=3, marker_size=8)
    monthly_figure.update_layout(
        xaxis_title="PO month", yaxis_title="Detected anomalies", margin=dict(t=55, b=10)
    )
    right.plotly_chart(monthly_figure, use_container_width=True)

    left, right = st.columns(2)
    supplier_summary = (
        anomalies_df.groupby("SUPPLIER").size().reset_index(name="ANOMALY_COUNT")
        .nlargest(10, "ANOMALY_COUNT")
    )
    supplier_figure = px.bar(
        supplier_summary,
        x="SUPPLIER",
        y="ANOMALY_COUNT",
        color="ANOMALY_COUNT",
        color_continuous_scale="Teal",
        title="Top suppliers by detected anomalies",
    )
    supplier_figure.update_layout(coloraxis_showscale=False, xaxis_title="Supplier", yaxis_title="Anomalies")
    left.plotly_chart(supplier_figure, use_container_width=True)

    material_summary = (
        anomalies_df.groupby("MATERIAL").size().reset_index(name="ANOMALY_COUNT")
        .nlargest(10, "ANOMALY_COUNT")
        .sort_values("ANOMALY_COUNT")
    )
    material_figure = px.bar(
        material_summary,
        x="ANOMALY_COUNT",
        y="MATERIAL",
        orientation="h",
        color="ANOMALY_COUNT",
        color_continuous_scale="Blues",
        title="Top materials by detected anomalies",
    )
    material_figure.update_layout(coloraxis_showscale=False, xaxis_title="Anomalies", yaxis_title="Material")
    right.plotly_chart(material_figure, use_container_width=True)

    if len(currencies_in_scope) > 1:
        leakage_summary = (
            filtered_df.groupby("LOCAL_CURRENCY")["PRICE_LEAKAGE_VALUE"]
            .sum().reset_index()
        )
        leakage_figure = px.bar(
            leakage_summary,
            x="LOCAL_CURRENCY",
            y="PRICE_LEAKAGE_VALUE",
            title="Potential price leakage by local currency",
            color="LOCAL_CURRENCY",
        )
        leakage_figure.update_layout(showlegend=False, xaxis_title="Currency", yaxis_title="Potential leakage")
        st.plotly_chart(leakage_figure, use_container_width=True)

with review_tab:
    st.subheader("Top model-detected anomalies")
    review_columns = [
        "PURCHASE_ORDER", "PO_ITEM", "PO_DATE", "COMPANY_CODE", "PLANT",
        "PURCHASING_ORG", "SUPPLIER", "MATERIAL", "LOCAL_CURRENCY",
        "UNIT_PRICE_LOCAL", "INFO_UNIT_PRICE", "HIST_AVG_UNIT_PRICE",
        "CROSS_SUPPLIER_6M_AVG_PRICE", "PRICE_LEAKAGE_VALUE",
        "ML_ANOMALY_SCORE", "ML_RISK_LEVEL", "ML_ANOMALY_REASON",
    ]
    available_columns = [c for c in review_columns if c in anomalies_df.columns]
    review_df = anomalies_df.sort_values("ML_ANOMALY_SCORE", ascending=False)[available_columns]
    st.dataframe(
        review_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PO_DATE": st.column_config.DateColumn("PO date", format="YYYY-MM-DD"),
            "ML_ANOMALY_SCORE": st.column_config.NumberColumn("Anomaly score", format="%.3f"),
            "UNIT_PRICE_LOCAL": st.column_config.NumberColumn("Actual unit price", format="%.2f"),
            "INFO_UNIT_PRICE": st.column_config.NumberColumn("Info unit price", format="%.2f"),
            "PRICE_LEAKAGE_VALUE": st.column_config.NumberColumn("Potential leakage", format="%.2f"),
        },
    )
    st.download_button(
        "Download filtered anomalies as CSV",
        review_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_po_price_anomalies.csv",
        mime="text/csv",
    )

with model_tab:
    st.subheader("Model and scoring context")
    model_name = source_df.get("MODEL_NAME", pd.Series(["PO_PRICE_KMEANS"])).dropna().iloc[0]
    model_version = source_df.get("MODEL_VERSION", pd.Series(["V1"])).dropna().iloc[0]
    threshold = source_df.get("ANOMALY_THRESHOLD", pd.Series([None])).dropna()
    threshold_text = f"{threshold.iloc[0]:.6f}" if not threshold.empty else "Not available"
    st.markdown(
        f"""
        - **Model:** {model_name}
        - **Version:** {model_version}
        - **Selected anomaly threshold:** {threshold_text}
        - **Algorithm:** KMeans clustering with distance-to-cluster-center scoring
        - **Feature preparation:** median imputation, vector assembly, and standard scaling
        - **Training period:** July 2022 through December 2025
        - **Testing and scoring period:** January through December 2026
        """
    )
    st.info(
        "An anomaly means unusual relative to learned price patterns. It does not by itself prove an error, policy violation, or fraud. Procurement review is required."
    )

st.caption(
    "Prepared by Balaguravareddy Duggireddy  |  Synthetic learning data  |  PySpark ML predictions"
)
