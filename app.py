"""
Contract Reconciliation & Analysis Tool - VISUAL DEMO
Risk Management department | fake dataset, static results.

Run locally :  streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Contract Reconciliation Tool",
    page_icon="📊",
    layout="wide",
)

NAVY = "#0B2545"
BLUE = "#13A8D8"
GREEN = "#2BB673"
ORANGE = "#F59E0B"
RED = "#E5484D"
PALETTE = [BLUE, NAVY, GREEN, ORANGE, RED, "#8B5CF6", "#64748B"]

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.6rem; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    div[data-testid="stMetric"] {{
        background: #F5F8FB; border: 1px solid #E2E8F0;
        border-left: 5px solid {BLUE};
        padding: 14px 16px; border-radius: 10px;
    }}
    .demo-banner {{
        background: #FFF7E6; border: 1px solid #FFD591; color: #8A5A00;
        padding: 8px 14px; border-radius: 8px; font-size: 0.85rem;
        margin-bottom: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Fake data (fixed seed -> always the same result)
# ----------------------------------------------------------------------------
COUNTRIES = ["Netherlands", "Belgium", "Germany", "France", "Spain", "Italy", "Poland"]
BRANDS = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E", "Brand F"]
TYPES = ["Passenger car", "SUV", "Van", "Electric", "Hybrid"]
PERIODS = ["2024-12", "2025-03", "2025-06", "2025-09"]


@st.cache_data
def load_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 2400
    country_w = [0.28, 0.17, 0.2, 0.14, 0.09, 0.07, 0.05]
    country = rng.choice(COUNTRIES, n, p=country_w)
    brand = rng.choice(BRANDS, n)
    vtype = rng.choice(TYPES, n, p=[0.3, 0.25, 0.15, 0.15, 0.15])

    base_value = rng.normal(34000, 8000, n).clip(14000, 90000)
    base_value *= np.where(vtype == "Electric", 1.25, 1.0)
    base_value *= np.where(vtype == "Van", 0.95, 1.0)
    duration = rng.choice([24, 36, 48, 60], n, p=[0.2, 0.4, 0.3, 0.1])
    mileage = rng.normal(28000, 9000, n).clip(8000, 70000)
    depreciation = (0.08 * duration / 12 + mileage / 400000
                    + rng.normal(0, 0.03, n)).clip(0.1, 0.65)
    residual = base_value * (1 - depreciation)
    profit = (base_value * 0.045 - depreciation * 900
              + rng.normal(0, 600, n))

    start = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1400, n), unit="D"
    )
    end = start + pd.to_timedelta(duration * 30.4, unit="D")

    df = pd.DataFrame(
        {
            "contract_id": [f"CT-{100000 + i}" for i in range(n)],
            "country": country,
            "brand": brand,
            "vehicle_type": vtype,
            "start_date": start.date,
            "end_date": end.date,
            "duration_months": duration,
            "mileage_km": mileage.round(0),
            "contract_value": base_value.round(0),
            "residual_value": residual.round(0),
            "depreciation_pct": (depreciation * 100).round(1),
            "profit": profit.round(0),
        }
    )
    return df


@st.cache_data
def load_reconciliation():
    """Static reconciliation result, identical on every run."""
    rng = np.random.default_rng(7)
    df = load_data()

    missing = df.sample(46, random_state=3).copy()
    missing["status"] = "Missing in newer period"
    missing["last_seen_period"] = "2025-03"
    missing["comment"] = rng.choice(
        ["Not found in new extract", "End date matches, ID absent",
         "Possible early termination"], len(missing))

    dup = df.sample(31, random_state=11).copy()
    dup["status"] = "Duplicate"
    dup["occurrences"] = rng.choice([2, 2, 2, 3], len(dup))
    dup["comment"] = rng.choice(
        ["Same ID + same end date", "Double entry, different system",
         "Re-booked after amendment"], len(dup))

    mismatch = df.sample(18, random_state=21).copy()
    mismatch["status"] = "End date mismatch"
    mismatch["comment"] = "Same ID, end date changed between periods"

    cols = ["contract_id", "country", "brand", "vehicle_type", "end_date",
            "contract_value", "status", "comment"]
    out = pd.concat([missing[cols], dup[cols], mismatch[cols]],
                    ignore_index=True)
    return out


df_all = load_data()
recon_all = load_reconciliation()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Contract Tool")
    st.caption("Risk Management · Reconciliation & Analysis")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔗 Correlation analysis", "🔍 Reconciliation"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Filters**")
    sel_countries = st.multiselect("Country entity", COUNTRIES, default=COUNTRIES)
    sel_brands = st.multiselect("Brand", BRANDS, default=BRANDS)
    sel_types = st.multiselect("Vehicle type", TYPES, default=TYPES)
    st.divider()
    st.caption("Dataset: 2,400 contracts · 4 periods · demo data")

df = df_all[
    df_all["country"].isin(sel_countries)
    & df_all["brand"].isin(sel_brands)
    & df_all["vehicle_type"].isin(sel_types)
]

st.markdown(
    '<div class="demo-banner">⚠️ Demo version – all figures come from a '
    "randomly generated dataset and do not reflect real company data.</div>",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No contract matches the selected filters.")
    st.stop()

# ----------------------------------------------------------------------------
# PAGE 1 - Overview
# ----------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("Portfolio overview")
    st.caption("Key indicators by country entity, brand and vehicle type")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active contracts", f"{len(df):,}".replace(",", " "), "+3.2% vs prev. period")
    c2.metric("Average contract value", f"€ {df.contract_value.mean():,.0f}".replace(",", " "), "-1.4%")
    c3.metric("Average profit / contract", f"€ {df.profit.mean():,.0f}".replace(",", " "), "+0.8%")
    c4.metric("Avg. depreciation", f"{df.depreciation_pct.mean():.1f} %", "+0.6 pt", delta_color="inverse")

    st.markdown("### Analysis by dimension")
    dim_label = st.selectbox(
        "Group by",
        ["Country entity", "Brand", "Vehicle type"],
    )
    dim = {"Country entity": "country", "Brand": "brand",
           "Vehicle type": "vehicle_type"}[dim_label]

    agg = (
        df.groupby(dim)
        .agg(contracts=("contract_id", "count"),
             avg_value=("contract_value", "mean"),
             avg_profit=("profit", "mean"))
        .reset_index()
        .sort_values("contracts", ascending=False)
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        fig = px.bar(agg, x=dim, y="contracts", title="Number of contracts",
                     color_discrete_sequence=[BLUE])
        fig.update_layout(xaxis_title=None, height=340)
        st.plotly_chart(fig, width="stretch")
    with g2:
        fig = px.bar(agg, x=dim, y="avg_value", title="Average value (€)",
                     color_discrete_sequence=[NAVY])
        fig.update_layout(xaxis_title=None, height=340)
        st.plotly_chart(fig, width="stretch")
    with g3:
        fig = px.bar(agg, x=dim, y="avg_profit", title="Average profit (€)",
                     color_discrete_sequence=[GREEN])
        fig.update_layout(xaxis_title=None, height=340)
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        agg.rename(columns={dim: dim_label, "contracts": "Contracts",
                            "avg_value": "Avg value (€)",
                            "avg_profit": "Avg profit (€)"})
        .style.format({"Avg value (€)": "{:,.0f}", "Avg profit (€)": "{:,.0f}"}),
        width="stretch", hide_index=True,
    )

# ----------------------------------------------------------------------------
# PAGE 2 - Correlation analysis
# ----------------------------------------------------------------------------
elif page == "🔗 Correlation analysis":
    st.title("Correlation analysis")
    st.caption("What drives depreciation and changes in vehicle valuation?")

    tab_a, tab_b, tab_c = st.tabs(
        ["By country", "Correlation matrix", "Explore two variables"]
    )

    with tab_a:
        st.markdown("#### Contracts, value and profit by country")
        by_c = (
            df.groupby("country")
            .agg(contracts=("contract_id", "count"),
                 total_value=("contract_value", "sum"),
                 avg_value=("contract_value", "mean"),
                 total_profit=("profit", "sum"),
                 avg_depr=("depreciation_pct", "mean"))
            .reset_index()
            .sort_values("contracts", ascending=False)
        )
        metric = st.radio(
            "Sort / display by",
            ["Contracts", "Value", "Profit"],
            horizontal=True,
        )
        col = {"Contracts": "contracts", "Value": "total_value",
               "Profit": "total_profit"}[metric]
        by_c = by_c.sort_values(col, ascending=False)

        l, r = st.columns([3, 2])
        with l:
            fig = px.bar(by_c, x="country", y=col, color="avg_depr",
                         color_continuous_scale="Blues",
                         labels={"avg_depr": "Avg depr. %", col: metric})
            fig.update_layout(height=380, xaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        with r:
            fig = px.scatter(by_c, x="total_value", y="total_profit",
                             size="contracts", color="country",
                             color_discrete_sequence=PALETTE,
                             labels={"total_value": "Total value (€)",
                                     "total_profit": "Total profit (€)"})
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.dataframe(
            by_c.rename(columns={
                "country": "Country", "contracts": "Contracts",
                "total_value": "Total value (€)", "avg_value": "Avg value (€)",
                "total_profit": "Total profit (€)", "avg_depr": "Avg depr. (%)"})
            .style.format({"Total value (€)": "{:,.0f}", "Avg value (€)": "{:,.0f}",
                           "Total profit (€)": "{:,.0f}", "Avg depr. (%)": "{:.1f}"}),
            width="stretch", hide_index=True,
        )

    with tab_b:
        st.markdown("#### Correlation table")
        num_cols = ["contract_value", "residual_value", "depreciation_pct",
                    "mileage_km", "duration_months", "profit"]
        nice = {"contract_value": "Contract value",
                "residual_value": "Residual value",
                "depreciation_pct": "Depreciation %",
                "mileage_km": "Mileage",
                "duration_months": "Duration",
                "profit": "Profit"}
        corr = df[num_cols].corr().rename(index=nice, columns=nice)

        l, r = st.columns([3, 2])
        with l:
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")
        with r:
            st.dataframe(
                corr.style.format("{:.2f}").background_gradient(
                    cmap="RdBu_r", vmin=-1, vmax=1),
                width="stretch", height=300,
            )
            st.info(
                "**Reading guide** – values close to ±1 indicate a strong "
                "relationship. Here, depreciation is mainly driven by "
                "duration and mileage."
            )

    with tab_c:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", list(nice.values()), index=3)
        y = c2.selectbox("Y axis", list(nice.values()), index=2)
        color = c3.selectbox("Color by", ["country", "brand", "vehicle_type"])
        inv = {v: k for k, v in nice.items()}
        sample = df.sample(min(len(df), 900), random_state=1)
        fig = px.scatter(sample, x=inv[x], y=inv[y], color=color,
                         trendline="ols", opacity=0.65,
                         color_discrete_sequence=PALETTE)
        fig.update_layout(height=480)
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# PAGE 3 - Reconciliation
# ----------------------------------------------------------------------------
else:
    st.title("Contract reconciliation")
    st.caption("Detect missing and duplicate contracts by comparing contract "
               "numbers and end dates between two periods")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
        period_a = c1.selectbox("Reference period", PERIODS, index=1)
        period_b = c2.selectbox("Compared period", PERIODS, index=2)
        scope = c3.multiselect("Country scope", COUNTRIES, default=COUNTRIES,
                               key="recon_scope")
        c4.write("")
        c4.write("")
        launch = c4.button("🔍 Launch search", type="primary",
                           width="stretch")

    if launch:
        bar = st.progress(0, text="Loading datasets…")
        steps = ["Loading datasets…", "Matching contract numbers…",
                 "Comparing end dates…", "Flagging anomalies…"]
        for i, s in enumerate(steps):
            time.sleep(0.45)
            bar.progress((i + 1) * 25, text=s)
        bar.empty()
        st.session_state["recon_done"] = True

    if not st.session_state.get("recon_done"):
        st.info("Select two periods and click **Launch search** to run the "
                "reconciliation.")
    else:
        res = recon_all[recon_all["country"].isin(scope)] if scope else recon_all
        n_miss = (res.status == "Missing in newer period").sum()
        n_dup = (res.status == "Duplicate").sum()
        n_mis = (res.status == "End date mismatch").sum()

        st.success(f"Reconciliation complete: {period_a} ➜ {period_b}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contracts compared", "2 400")
        m2.metric("Missing contracts", n_miss)
        m3.metric("Duplicate contracts", n_dup)
        m4.metric("End date mismatches", n_mis)

        g1, g2 = st.columns(2)
        with g1:
            by_cs = res.groupby(["country", "status"]).size().reset_index(name="count")
            fig = px.bar(by_cs, x="country", y="count", color="status",
                         barmode="stack", title="Anomalies by country",
                         color_discrete_map={
                             "Missing in newer period": RED,
                             "Duplicate": ORANGE,
                             "End date mismatch": BLUE})
            fig.update_layout(height=400, xaxis_title=None,
                              legend_title=None, legend_orientation="h",
                              legend_y=-0.2)
            st.plotly_chart(fig, width="stretch")
        with g2:
            wf = go.Figure(go.Waterfall(
                x=[period_a, "Missing", "Duplicates", "New contracts", period_b],
                measure=["absolute", "relative", "relative", "relative", "total"],
                y=[2400, -int(n_miss), int(n_dup), 84, 0],
                connector={"line": {"color": "#94A3B8"}},
                decreasing={"marker": {"color": RED}},
                increasing={"marker": {"color": GREEN}},
                totals={"marker": {"color": NAVY}},
            ))
            wf.update_layout(title="Contract count bridge", height=400,
                             yaxis_range=[2300, 2500])
            st.plotly_chart(wf, width="stretch")

        st.markdown("### Detail of flagged contracts")
        f1, f2 = st.columns([2, 3])
        status_filter = f1.multiselect(
            "Status",
            ["Missing in newer period", "Duplicate", "End date mismatch"],
            default=["Missing in newer period", "Duplicate", "End date mismatch"],
        )
        view = res[res.status.isin(status_filter)]
        st.dataframe(
            view.rename(columns={
                "contract_id": "Contract no.", "country": "Country",
                "brand": "Brand", "vehicle_type": "Type",
                "end_date": "End date", "contract_value": "Value (€)",
                "status": "Status", "comment": "Comment"})
            .style.format({"Value (€)": "{:,.0f}"}),
            width="stretch", hide_index=True, height=360,
        )
        st.download_button(
            "⬇️ Export results (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="reconciliation_results.csv",
            mime="text/csv",
        )
