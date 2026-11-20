"""DSA Tool: the screen you interact with.

Start it with:   .venv/bin/streamlit run app.py
It opens in your browser at http://localhost:8501 and runs only on this computer.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dsa.data import fx_share as fxs
from dsa.data import maturity, oil_trade
from dsa.data import overrides as country_overrides
from dsa.data.weo import COUNTRY_SERIES, DERIVED_SERIES, WORLD_SERIES, get_country, list_countries
from dsa.engine import baseline_from_weo, debt_path, historical_drivers, historical_stats
from dsa.fanchart import fan_chart
from dsa.shocks import PRESETS, CustomShock, custom, historical_scenario, preset

OIL_EXPORTERS = {
    "AGO", "ARE", "AZE", "BHR", "BRN", "COG", "DZA", "ECU", "GAB", "GNQ", "GUY", "IRN", "IRQ",
    "KAZ", "KWT", "LBY", "NGA", "NOR", "OMN", "QAT", "RUS", "SAU", "SSD", "TCD", "TKM", "TTO", "VEN",
}
TABLE_ROWS = {
    "debt": "Gross public debt",
    "change_in_debt": "Change in debt",
    "identified_flows": "Identified debt-creating flows",
    "primary_deficit": "  Primary deficit",
    "automatic_debt_dynamics": "  Automatic debt dynamics",
    "real_interest_rate": "    of which: real interest rate",
    "real_gdp_growth": "    of which: real GDP growth",
    "exchange_rate": "    of which: exchange rate",
    "other_flows": "  Residual / other flows",
    "gross_financing_need": "Gross financing need",
    "interest_bill": "  of which: interest",
    "amortisation": "  of which: amortisation",
    "debt_stabilising_pb": "Debt-stabilising primary balance",
}
COLOURS = {"baseline": "#1f4e79", "custom": "#c0392b", "historical": "#7f8c8d"}

st.set_page_config(page_title="DSA Tool", layout="wide")


@st.cache_data(show_spinner="Downloading IMF WEO data…")
def load(iso3: str, refresh_token: int):
    return get_country(iso3, refresh=refresh_token > 0)


COVERED_PATH = Path(__file__).parent / "dsa" / "data" / "countries_covered.csv"


@st.cache_data
def countries() -> dict[str, str]:
    """Countries shown in the dropdown: those listed in dsa/data/countries_covered.csv.

    To add or remove a country, add or delete its row in that file. If the file
    is missing, every IMF country is shown.
    """
    try:
        names = list_countries()
    except Exception:
        names = {}
    if COVERED_PATH.exists():
        covered = pd.read_csv(COVERED_PATH)
        names = {iso: names.get(iso, name).strip() for iso, name in zip(covered["iso3"], covered["name"])}
    return dict(sorted(names.items(), key=lambda kv: kv[1]))


@st.cache_data(show_spinner="Simulating 10,000 debt paths…")
def cached_fan(_baseline, history, key):
    return fan_chart(_baseline, history)


def describe(info: dict, field: str) -> str:
    """Caption under an input: where its starting value came from."""
    label = "Article IV" if info.get("method") == "override" else "Estimate"
    as_of = info.get("as_of")
    has_date = isinstance(as_of, str) and as_of.strip() != "" and as_of.lower() != "nan"
    text = f"{label}: {info[field]:.1f}%, {info['source']}" + (f", {as_of}" if has_date else "")
    if info.get("method") == "override":
        auto = info.get("automatic_estimate")
        return text + (f". Automatic estimate was {auto:.1f}%." if auto is not None else ".")
    return text + f" (quality: {info['quality']})"


# ---------------------------------------------------------------- sidebar: inputs
names = countries()
codes = list(names) or ["BRA"]
default = codes.index("BRA") if "BRA" in codes else 0
st.sidebar.title("DSA Tool")

# The shock controls reset to zero whenever you pick another country: each
# control's key includes a counter that goes up on every country change.
if "shock_round" not in st.session_state:
    st.session_state.shock_round = 0


def _new_country() -> None:
    st.session_state.shock_round += 1


def shock_key(name: str, *extra) -> str:
    return "_".join(["shock", name, str(st.session_state.shock_round), *map(str, extra)])


iso3 = st.sidebar.selectbox(
    "Country", codes, index=default, format_func=lambda c: f"{names.get(c, c)} ({c})", on_change=_new_country
)
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if st.sidebar.button("Re-download latest WEO data"):
    st.session_state.refresh += 1

try:
    data = load(iso3, st.session_state.refresh)
except Exception as exc:  # no data for this code, or no internet
    st.error(f"Could not load WEO data for {iso3}: {exc}")
    st.stop()
table, last = data.table, data.last_actual_year

st.sidebar.subheader("Country assumptions")
fx_info = fxs.lookup(iso3)
fx_default = float(fx_info["fx_share"]) if fx_info["fx_share"] is not None else 0.0
fx = st.sidebar.number_input("Foreign-currency share of debt (%)", 0.0, 100.0, fx_default, 1.0, key=f"fx_{iso3}")
if fx_info["fx_share"] is None:
    st.sidebar.warning("No estimate found for this country. Enter the share from the latest Article IV DSA.")
else:
    st.sidebar.caption(describe(fx_info, "fx_share"))

mat_info = maturity.lookup(iso3)
repricing = st.sidebar.slider(
    "Share of debt maturing or repricing each year (%)", 1, 100, int(round(mat_info["maturing_share"])), 1,
    key=f"mat_{iso3}",
    help="Sets how fast an interest rate shock feeds into the average interest rate, and the amortisation in gross financing needs. 25% means an average maturity of about 4 years.",
)
mat_note = describe(mat_info, "maturing_share")
if mat_info["quality"] == "none":
    st.sidebar.warning("No maturity data for this country: using 25%. Check the Article IV for debt falling due within a year.")
else:
    st.sidebar.caption(mat_note)
override_r = st.sidebar.checkbox(
    "Override effective interest rate",
    help="WEO gives net interest (interest paid minus interest received), which understates the rate for countries with large assets.",
)
overrides = {}
if override_r:
    overrides["effective_rate"] = st.sidebar.number_input("Effective interest rate (%)", -5.0, 50.0, float(table.loc[last, "effective_rate"]), 0.1)
horizon = st.sidebar.slider("Projection horizon (years)", 5, 10, 10)

st.sidebar.subheader("Your shocks")
oil_info = oil_trade.lookup(iso3)
net_oil = oil_info["net_oil_exports"]
if net_oil is not None:  # default role from the country's actual fuel trade
    role_index = 0 if net_oil > 1 else 1 if net_oil < -0.5 else 2
else:
    role_index = 0 if iso3 in OIL_EXPORTERS else 1
oil_role = st.sidebar.radio("Oil (fiscal channel)", ["exporter", "importer", "none"], index=role_index, horizontal=True, key=f"oil_{iso3}")
with st.sidebar.expander("Growth, rates, primary balance", expanded=True):
    g_pp = st.slider("Real growth (pp)", -10.0, 5.0, 0.0, 0.5, key=shock_key("g_pp"))
    g_yrs = st.slider("Growth shock years", 1, horizon, (2, 3), key=shock_key("g_yrs", horizon))
    r_bp = st.slider("Market interest rates (bp)", -300, 1000, 0, 25, key=shock_key("r_bp"))
    r_from = st.slider("Rate shock starts in year", 1, horizon, 2, key=shock_key("r_from", horizon))
    pb_pp = st.slider("Primary balance (pp of GDP)", -8.0, 4.0, 0.0, 0.25, key=shock_key("pb_pp"))
    pb_yrs = st.slider("Primary balance shock years", 1, horizon, (2, 3), key=shock_key("pb_yrs", horizon))
with st.sidebar.expander("Exchange rate and oil", expanded=True):
    dep = st.slider("One-off depreciation (%)", -20, 80, 0, 5, key=shock_key("dep"))
    dep_yr = st.slider("Depreciation in year", 1, horizon, 2, key=shock_key("dep_yr", horizon))
    oil_pct = st.slider("Oil price change (%)", -70, 100, 0, 5, key=shock_key("oil_pct"))
    oil_yrs = st.slider("Oil shock years", 1, horizon, (2, 3), key=shock_key("oil_yrs", horizon))
    default_sens = 0.5 if oil_role == "exporter" else 0.9
    oil_sens = st.number_input(
        "Primary balance impact per 10% oil move (pp of GDP)", 0.0, 10.0, default_sens, 0.1, key=shock_key("oil_sens", oil_role),
        help="IMF default (MAC SRDSF Table 5): exporters lose 0.5pp of revenue per 10% fall; importers spend 0.9pp more per 10% rise.",
    )
    oil_growth = st.number_input("Growth impact per 10% oil move (pp)", 0.0, 5.0, 0.0, 0.1, key=shock_key("oil_growth"), help="The IMF paper gives no default; set your own.")
with st.sidebar.expander("Current account and external financing", expanded=True):
    ca_pp = st.slider("Current account (pp of GDP)", -10.0, 5.0, 0.0, 0.25, key=shock_key("ca_pp"), help="Negative = a wider deficit (or smaller surplus) than the WEO baseline.")
    ca_yrs = st.slider("Current account shock years", 1, horizon, (2, 3), key=shock_key("ca_yrs", horizon))
    oil_ca = st.checkbox("Oil price also moves the current account", True, disabled=net_oil is None, key=shock_key("oil_ca"))
    if net_oil is not None:
        st.caption(
            f"Net fuel exports: {net_oil:+.1f}% of GDP ({oil_info['source']}, {oil_info['as_of']}). "
            f"A 10% oil rise moves the current account by {net_oil / 10:+.2f}pp of GDP."
        )
    dep_per_ca = st.number_input(
        "Depreciation per 1pp of GDP current account worsening (%)", 0.0, 30.0, 5.0, 0.5, key=shock_key("dep_per_ca"),
        help="How much the currency weakens to close a wider current account gap. No IMF default: your judgement (lower if reserves or official financing cover the gap).",
    )
    ext_stop = st.checkbox(
        "Foreign financing stop", False, key=shock_key("ext_stop"),
        help="IMF MAC SRDSF GFN module stress: foreign private investors roll over only part of their maturing holdings; the rest is replaced by one-year domestic borrowing at a higher rate.",
    )
    fp_override = country_overrides.get(iso3, "foreign_private_share")
    fp_share = st.number_input(
        "Debt held by foreign private investors (% of total)", 0.0, 100.0, fp_override["value"] if fp_override else 0.0, 1.0, key=f"fp_{iso3}",
    )
    if fp_override:
        st.caption(f"Article IV: {fp_override['source']}")
    elif ext_stop:
        st.caption("No figure on file for this country: enter it from the Article IV (debt by holder chart).")
    rollover = st.slider("Foreign rollover rate in stress (%)", 0, 100, 67, 1, key=shock_key("rollover"), help="IMF default: 67%.")
    stop_yrs = st.slider("Financing stop years", 1, horizon, (2, 3), key=shock_key("stop_yrs", horizon))
    premium = st.slider("Extra cost of replacement borrowing (bp)", 0, 1500, 200, 25, key=shock_key("premium"))
with st.sidebar.expander("Other"):
    cl = st.slider("Contingent liability (% of GDP, one-off)", 0.0, 30.0, 0.0, 0.5, key=shock_key("cl"))
    cl_yr = st.slider("Contingent liability in year", 1, horizon, 2, key=shock_key("cl_yr", horizon))
    interactions = st.checkbox("Apply IMF knock-on effects", True, key=shock_key("interactions"), help="Growth shocks lower inflation and raise spending/GDP; depreciation raises inflation; weaker primary balances raise borrowing costs (LIC-DSF 2018, Table 8).")

# ---------------------------------------------------------------- run the model
baseline = baseline_from_weo(table, last, fx_share=fx, horizon=horizon, repricing_share=repricing, overrides=overrides)
shock = CustomShock(
    growth_pp=g_pp, growth_years=g_yrs, interest_bp=r_bp, interest_years=(r_from, horizon),
    primary_balance_pp=pb_pp, pb_years=pb_yrs, depreciation_pct=dep, depreciation_year=dep_yr,
    oil_price_pct=oil_pct, oil_years=oil_yrs, oil_role=oil_role, oil_fiscal_pp_per_10pct=oil_sens,
    oil_growth_pp_per_10pct=oil_growth, contingent_liability_pp=cl, contingent_year=cl_yr, interactions=interactions,
    current_account_pp=ca_pp, ca_years=ca_yrs, oil_moves_current_account=oil_ca, net_oil_exports=net_oil,
    depreciation_per_ca_pp=dep_per_ca, external_stop=ext_stop, stop_years=stop_yrs, foreign_private_share=fp_share,
    foreign_rollover_pct=rollover, replacement_premium_bp=premium,
)
shocked = custom(baseline, shock)
base_out, shock_out = debt_path(baseline), debt_path(shocked)
stats = historical_stats(table, last)
history = historical_drivers(table, last)
has_shock = any([g_pp, r_bp, pb_pp, dep, oil_pct, cl, ca_pp, ext_stop and fp_share])

hist_years = table.loc[last - 9 : last, "debt"]


def yr_label(y: int) -> str:
    return f"{y}" if y <= last else f"{y}p"


# ---------------------------------------------------------------- header
st.title(f"{data.name}: debt sustainability analysis")
t_end = base_out.index[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Debt / GDP, {last}", f"{table.loc[last, 'debt']:.1f}%")
c2.metric(f"Baseline, {t_end}", f"{base_out['debt'].iloc[-1]:.1f}%", f"{base_out['debt'].iloc[-1] - table.loc[last, 'debt']:+.1f}pp", delta_color="inverse")
c3.metric(f"Your scenario, {t_end}", f"{shock_out['debt'].iloc[-1]:.1f}%", f"{shock_out['debt'].iloc[-1] - base_out['debt'].iloc[-1]:+.1f}pp vs baseline", delta_color="inverse")
gfn_base = base_out["gross_financing_need"].iloc[:5].mean()
gfn_shock = shock_out["gross_financing_need"].iloc[:5].mean()
c4.metric(
    "Average GFN, first 5 years (your scenario)", f"{gfn_shock:.1f}% of GDP",
    f"{gfn_shock - gfn_base:+.1f}pp vs baseline ({gfn_base:.1f}%)", delta_color="inverse",
)
st.caption(f"Source: IMF World Economic Outlook, downloaded {data.downloaded}. Actual data to {last}; later years are WEO projections, extended at constant drivers beyond the WEO horizon.")
for note in baseline.notes:
    st.info(note)

tab_paths, tab_drivers, tab_stress, tab_fan, tab_data = st.tabs(["Debt paths", "Drivers", "Stress tests", "Fan chart", "Data"])

# ---------------------------------------------------------------- debt paths
with tab_paths:
    fig = go.Figure()
    fig.add_scatter(x=hist_years.index, y=hist_years.values, name="Actual", line=dict(color="black"))
    joined = lambda out: pd.concat([pd.Series({last: table.loc[last, "debt"]}), out["debt"]])
    fig.add_scatter(x=joined(base_out).index, y=joined(base_out).values, name="Baseline (WEO)", line=dict(color=COLOURS["baseline"], width=3))
    if has_shock:
        fig.add_scatter(x=joined(shock_out).index, y=joined(shock_out).values, name="Your scenario", line=dict(color=COLOURS["custom"], width=3))
    hs = debt_path(historical_scenario(baseline, stats))
    fig.add_scatter(x=joined(hs).index, y=joined(hs).values, name="Historical scenario", line=dict(color=COLOURS["historical"], dash="dot"))
    fig.add_vline(x=last + 0.5, line=dict(color="grey", dash="dash"))
    fig.update_layout(yaxis_title="Gross public debt, % of GDP", height=480, legend=dict(orientation="h"), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Historical scenario: growth, primary balance, inflation, interest rate and depreciation at their 10-year averages (IMF A1 test).")

    st.subheader("Key assumptions")
    show = pd.DataFrame(
        {
            "Real growth (%)": baseline.real_growth, "Your scenario: growth": shocked.real_growth,
            "Deflator inflation (%)": baseline.deflator_growth, "Your scenario: inflation": shocked.deflator_growth,
            "Effective interest rate (%)": baseline.effective_rate, "Your scenario: interest rate": shocked.effective_rate,
            "Primary balance (% GDP)": baseline.primary_balance, "Your scenario: primary balance": shocked.primary_balance,
            "Depreciation (%)": baseline.fx_depreciation, "Your scenario: depreciation": shocked.fx_depreciation,
            "Current account (% GDP)": baseline.current_account if baseline.current_account is not None else pd.Series(dtype=float),
            "Your scenario: current account": shocked.current_account if shocked.current_account is not None else pd.Series(dtype=float),
            "Gross financing need (% GDP)": base_out["gross_financing_need"], "Your scenario: gross financing need": shock_out["gross_financing_need"],
        }
    ).T
    show.columns = [yr_label(c) for c in show.columns]
    st.dataframe(show.round(1), width="stretch")

# ---------------------------------------------------------------- drivers
with tab_drivers:
    which = st.radio("Scenario", ["Baseline", "Your scenario"], horizontal=True)
    out = base_out if which == "Baseline" else shock_out
    fig = go.Figure()
    for col, label in [("primary_deficit", "Primary deficit"), ("real_interest_rate", "Real interest rate"), ("real_gdp_growth", "Real GDP growth"), ("exchange_rate", "Exchange rate"), ("other_flows", "Residual / other")]:
        fig.add_bar(x=out.index, y=out[col], name=label)
    fig.add_scatter(x=out.index, y=out["change_in_debt"], name="Change in debt", mode="markers", marker=dict(color="black", size=10, symbol="diamond"))
    fig.update_layout(barmode="relative", yaxis_title="Contribution to change in debt, pp of GDP", height=450, legend=dict(orientation="h"), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.subheader("Debt decomposition (IMF Table A2 layout)")
    tbl = out[list(TABLE_ROWS)].rename(columns=TABLE_ROWS).T
    tbl.columns = [yr_label(c) for c in tbl.columns]
    st.dataframe(tbl.round(1), width="stretch")
    st.caption("Automatic debt dynamics = [r − π(1+g) − g + a·e(1+r)] / (1+g+π+gπ) × previous debt (IMF MAC SRDSF 2021, Table A2 note 5). The residual carries the gap between WEO debt projections and the identified flows. Gross financing need ≈ primary deficit + interest + the maturing share of debt.")

# ---------------------------------------------------------------- stress tests
with tab_stress:
    st.write("IMF standard stress tests, applied to the baseline in projection years 2 and 3 (10-year history for averages and standard deviations).")
    results = {"Baseline": base_out, "Historical scenario": hs}
    if has_shock:
        results["Your scenario"] = shock_out
    for name, label in PRESETS.items():
        results[label] = debt_path(preset(name, baseline, stats))
    fig = go.Figure()
    for label, out in results.items():
        style = dict(width=3) if label == "Baseline" else dict(width=1.5)
        fig.add_scatter(x=out.index, y=out["debt"], name=label, line=style)
    fig.update_layout(yaxis_title="Debt, % of GDP", height=480, legend=dict(orientation="v"), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    y5 = base_out.index[min(4, len(base_out) - 1)]
    summary = pd.DataFrame(
        {
            f"Debt {y5}": {k: v.loc[y5, "debt"] for k, v in results.items()},
            f"Debt {t_end}": {k: v["debt"].iloc[-1] for k, v in results.items()},
            "Peak debt": {k: v["debt"].max() for k, v in results.items()},
            "Peak GFN": {k: v["gross_financing_need"].max() for k, v in results.items()},
        }
    )
    st.dataframe(summary.round(1), width="stretch")
    st.caption("10-year historical statistics used for the 1 SD shocks:")
    st.dataframe(stats.round(2).T, width="stretch")

# ---------------------------------------------------------------- fan chart
with tab_fan:
    if len(history) < 6:
        st.warning("Not enough complete history to build a fan chart for this country.")
    else:
        key = (iso3, fx, repricing, tuple(overrides.items()), data.downloaded)
        fan = cached_fan(baseline, history, key)
        fig = go.Figure()
        x = [last] + fan.years
        start = table.loc[last, "debt"]
        bands = [("p5", "p95", "5th–95th", 0.15), ("p25", "p75", "25th–75th", 0.3)]
        for lo, hi, label, alpha in bands:
            fig.add_scatter(x=x, y=[start] + list(fan.standard[hi]), line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=x, y=[start] + list(fan.standard[lo]), fill="tonexty", fillcolor=f"rgba(31,78,121,{alpha})", line=dict(width=0), name=label)
        fig.add_scatter(x=x, y=[start] + list(fan.standard["p50"]), name="Median", line=dict(color=COLOURS["baseline"], dash="dot"))
        fig.add_scatter(x=x, y=[start] + list(fan.baseline), name="Baseline", line=dict(color="black", width=3))
        fig.add_scatter(x=x, y=[start] + list(fan.historical["p20"]), name="20th percentile, historical fan", line=dict(color="#e67e22", dash="dash"))
        fig.update_layout(yaxis_title="Debt, % of GDP", height=480, legend=dict(orientation="h"), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Fan width (5th–95th, final year)", f"{fan.width:.1f}pp")
        m2.metric("Probability debt does not stabilise", f"{fan.prob_not_stabilising:.0f}%")
        m3.metric("Realism flag", "Baseline below 20th pct" if fan.baseline_below_hist_p20 else "OK")
        st.caption(
            f"Method (MAC SRDSF 2021, Annex VI): {fan.draws:,} paths from 2-year blocks of the country's own historical surprises, {fan.history_years[0]}–{fan.history_years[1]}. "
            "The realism flag compares the baseline with the 20th percentile of a history-only fan. Not replicated: starting-debt uncertainty and the IMF's skewed 'realism-adjusted' fan."
        )

# ---------------------------------------------------------------- data + export
with tab_data:
    labels = {name: desc for _, (name, desc) in {**COUNTRY_SERIES, **WORLD_SERIES}.items()} | DERIVED_SERIES
    raw = table.loc[table.index >= last - 15].T
    raw.insert(0, "Description", [labels.get(i, i) for i in raw.index])
    raw.columns = ["Description"] + [yr_label(c) for c in raw.columns[1:]]
    st.dataframe(raw.round(2), width="stretch")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        base_out.T.to_excel(xl, sheet_name="Baseline")
        shock_out.T.to_excel(xl, sheet_name="Your scenario")
        summary.to_excel(xl, sheet_name="Stress tests")
        raw.to_excel(xl, sheet_name="WEO data")
        pd.DataFrame({"setting": ["country", "fx_share", "repricing_share", "shock"], "value": [iso3, fx, repricing, str(shock)]}).to_excel(xl, sheet_name="Settings", index=False)
    st.download_button("Download everything as Excel", buf.getvalue(), f"DSA_{iso3}_{data.downloaded}.xlsx")
