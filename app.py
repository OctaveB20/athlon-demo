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


def fmt(x) -> str:
    """Thousands separated by a space, e.g. 400 000."""
    return f"{int(x):,}".replace(",", " ")


@st.cache_resource
def load_data() -> pd.DataFrame:
    """Fake portfolio: uneven markets, skewed values, outliers, noise."""
    rng = np.random.default_rng(2024)
    n = 400_000

    # --- markets / brands: deliberately uneven ---------------------------
    cw = np.array([0.263, 0.158, 0.221, 0.117, 0.083, 0.094, 0.064])
    country = rng.choice(COUNTRIES, n, p=cw / cw.sum())
    bw = np.array([0.27, 0.19, 0.16, 0.14, 0.13, 0.11])
    brand = rng.choice(BRANDS, n, p=bw / bw.sum())

    # --- vehicle type & duration mix differ by country --------------------
    base_type = np.array([0.34, 0.22, 0.14, 0.12, 0.18])
    base_dur = np.array([0.18, 0.41, 0.28, 0.13])
    DURS = np.array([24, 36, 48, 60])
    vtype = np.empty(n, dtype=object)
    duration = np.empty(n, dtype=int)
    for c in COUNTRIES:
        m = country == c
        vtype[m] = rng.choice(TYPES, m.sum(), p=rng.dirichlet(base_type * 35))
        duration[m] = rng.choice(DURS, m.sum(), p=rng.dirichlet(base_dur * 30))

    # --- group effects ----------------------------------------------------
    c_val = dict(zip(COUNTRIES, [1.04, 1.02, 1.12, 0.97, 0.91, 0.94, 0.86]))
    c_km = dict(zip(COUNTRIES, [0.92, 0.88, 1.10, 1.00, 1.05, 0.95, 1.18]))
    c_dep = dict(zip(COUNTRIES, [-0.010, -0.005, -0.015, 0.008, 0.021, 0.014, 0.030]))
    c_prof = dict(zip(COUNTRIES, [140, 60, 210, -40, -120, -70, -160]))
    b_val = dict(zip(BRANDS, [1.18, 0.96, 1.05, 0.88, 1.32, 0.80]))
    b_dep = dict(zip(BRANDS, [-0.020, 0.010, -0.005, 0.025, -0.035, 0.030]))
    t_val = {"Passenger car": 1.0, "SUV": 1.14, "Van": 0.95,
             "Electric": 1.27, "Hybrid": 1.08}
    t_km = {"Passenger car": 1.0, "SUV": 0.97, "Van": 1.28,
            "Electric": 0.84, "Hybrid": 0.95}
    t_dep = {"Passenger car": 0.0, "SUV": -0.005, "Van": 0.015,
             "Electric": 0.045, "Hybrid": -0.010}

    ser = lambda a: pd.Series(a)
    cs, bs, ts = ser(country), ser(brand), ser(vtype)

    # --- values: right-skewed, with a few very expensive vehicles ---------
    value = rng.lognormal(np.log(30500), 0.27, n)
    value *= cs.map(c_val).to_numpy() * bs.map(b_val).to_numpy() * ts.map(t_val).to_numpy()
    value *= rng.normal(1, 0.04, n)
    value = value.clip(11000, 135000)

    # --- mileage: gamma distributed, depends on country / type ------------
    yearly_km = rng.gamma(6.0, 4300, n) * cs.map(c_km).to_numpy() * ts.map(t_km).to_numpy()
    mileage = (yearly_km * duration / 12).clip(5000, 260000)

    # --- depreciation: duration + mileage + group effects + noise ---------
    dep = (0.08 + 0.065 * duration / 12 + mileage / 1.0e6
           + cs.map(c_dep).to_numpy() + bs.map(b_dep).to_numpy()
           + ts.map(t_dep).to_numpy()
           + rng.normal(0, 0.045, n))
    shock = rng.random(n) < 0.02                      # market shock / damage
    dep = np.where(shock, dep + rng.uniform(0.05, 0.14, n), dep)
    dep = dep.clip(0.08, 0.80)
    residual = value * (1 - dep)

    # --- profit: heavy tails, some losses ---------------------------------
    profit = (value * 0.045 - dep * value * 0.07 + 150 * duration / 12
              + cs.map(c_prof).to_numpy()
              + rng.standard_t(4, n) * 620)

    # --- dates: more recent contracts, small end-date drift ---------------
    days = (rng.beta(2.2, 1.6, n) * 2200).astype(int)
    start = pd.Timestamp("2019-06-01") + pd.to_timedelta(days, unit="D")
    drift = rng.integers(-45, 46, n)
    end = start + pd.to_timedelta(duration * 30.44 + drift, unit="D")

    ids = 5_000_000 + np.cumsum(rng.integers(1, 40, n))   # irregular IDs
    return pd.DataFrame(
        {
            "contract_id": ("CT-" + pd.Series(ids).astype(str)).to_numpy(),
            "country": country,
            "brand": brand,
            "vehicle_type": vtype,
            "start_date": start.date,
            "end_date": end.date,
            "duration_months": duration,
            "mileage_km": mileage.round(0),
            "contract_value": value.round(0),
            "residual_value": residual.round(0),
            "depreciation_pct": (dep * 100).round(1),
            "profit": profit.round(0),
        }
    )


@st.cache_resource
def load_reconciliation():
    """Static reconciliation result, identical on every run."""
    rng = np.random.default_rng(7)
    df = load_data()
    n_total = len(df)

    # anomalies are not spread evenly: some entities are much worse than others
    country_f = pd.Series(rng.uniform(0.5, 1.9, len(COUNTRIES)), index=COUNTRIES)
    w = df["country"].map(country_f).to_numpy()

    missing = df.sample(int(n_total * 0.0031), weights=w, random_state=3).copy()
    missing["status"] = "Missing in newer period"
    missing["comment"] = rng.choice(
        ["Not found in new extract", "End date matches, ID absent",
         "Possible early termination"], len(missing), p=[0.5, 0.3, 0.2])

    dup = df.sample(int(n_total * 0.0017), weights=w ** 1.5, random_state=11).copy()
    dup["status"] = "Duplicate"
    dup["comment"] = rng.choice(
        ["Same ID + same end date", "Double entry, different system",
         "Re-booked after amendment"], len(dup), p=[0.55, 0.25, 0.20])

    mismatch = df.sample(int(n_total * 0.0011), weights=w[::-1].copy(), random_state=21).copy()
    mismatch["status"] = "End date mismatch"
    mismatch["comment"] = "Same ID, end date changed between periods"

    cols = ["contract_id", "country", "brand", "vehicle_type", "end_date",
            "contract_value", "status", "comment"]
    out = pd.concat([missing[cols], dup[cols], mismatch[cols]],
                    ignore_index=True)
    return out.sample(frac=1, random_state=5).reset_index(drop=True)


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
    st.caption(f"Dataset: {fmt(len(df_all))} contracts · 4 periods · demo data")

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
                corr.style.format("{:.2f}"),
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
        compared = int(df_all["country"].isin(scope).sum()) if scope else len(df_all)
        n_new = int(compared * 0.012)
        n_miss = (res.status == "Missing in newer period").sum()
        n_dup = (res.status == "Duplicate").sum()
        n_mis = (res.status == "End date mismatch").sum()

        st.success(f"Reconciliation complete: {period_a} ➜ {period_b}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contracts compared", fmt(compared))
        m2.metric("Missing contracts", fmt(n_miss))
        m3.metric("Duplicate contracts", fmt(n_dup))
        m4.metric("End date mismatches", fmt(n_mis))

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
                y=[compared, -int(n_miss), int(n_dup), n_new, 0],
                connector={"line": {"color": "#94A3B8"}},
                decreasing={"marker": {"color": RED}},
                increasing={"marker": {"color": GREEN}},
                totals={"marker": {"color": NAVY}},
            ))
            wf.update_layout(title="Contract count bridge", height=400,
                             yaxis_range=[compared - 3 * n_new, compared + 1.4 * n_new])
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
