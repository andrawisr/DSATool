"""Estimate the share of each country's government debt that falls due within a year.

In plain English: gross financing needs depend on how much debt has to be
repaid and refinanced each year, and an interest rate shock reaches the
average interest rate faster when more debt rolls over. The app's "share of
debt maturing or repricing each year" setting needs a country-specific
starting value. We estimate it in this order:

  1. World Bank Quarterly Public Sector Debt (QPSD): debt with payment due
     within 12 months (short-term debt plus long-term debt falling due), as a
     share of total debt, averaged over the last four quarters. Covers all
     debt: domestic and external, bonds and loans. Quality: high.
  2. External debt repayments (World Bank International Debt Statistics),
     plus a typical domestic rollover rate from comparable countries,
     weighted by the foreign-currency share. Principal repayments are used
     when the IDS service is available (medium quality); otherwise total
     debt service minus an assumed interest rate (low quality).
  3. Otherwise: 25%, flagged as a default.

Results outside 2-60% are treated as reporting problems and skipped.
Run `python -m dsa.data.maturity` to rebuild the table from fresh data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from . import fx_share
from .fx_share import MIN_COVERAGE, OLDEST_QUARTER, SECTORS, _wb_series, _weo_debt_levels

TABLE_PATH = Path(__file__).with_name("maturity_estimates.csv")

DEFAULT_SHARE = 25.0
PLAUSIBLE = (2.0, 60.0)
ASSUMED_EXTERNAL_INTEREST = 3.0  # % a year, used only when principal repayments aren't available
ADVANCED = fx_share.EURO_AREA | fx_share.OWN_CURRENCY_ADVANCED


def _qpsd_due_within_year(weo: pd.DataFrame) -> dict:
    frames = []
    for sector in SECTORS:
        for ind, kind in [
            (f"DP.DOD.DSTC.CR.{sector}", "short_term"),
            (f"DP.DOD.DLTC.CR.L1.{sector}", "long_term_due"),
            (f"DP.DOD.DECT.CR.{sector}", "total"),
        ]:
            frames.append(_wb_series(ind, source=20, dates=f"{OLDEST_QUARTER}:2030Q4").assign(sector=sector, kind=kind))
    q = pd.concat(frames)
    w = q.pivot_table(index=["iso3", "sector", "date"], columns="kind", values="value").reset_index()
    w.columns.name = None
    w = w.dropna(subset=["short_term", "long_term_due", "total"])
    w = w[(w["total"] > 0) & (w["date"] >= "2022Q1")]
    w["share"] = 100 * (w["short_term"] + w["long_term_due"]) / w["total"]
    w["year"] = w["date"].str[:4].astype(int)
    w = w.merge(weo, on=["iso3", "year"], how="left")

    found = {}
    for sector in SECTORS:  # general government first, then narrower coverage
        rows = w[w["sector"] == sector].sort_values("date")
        for iso3, grp in rows.groupby("iso3"):
            if iso3 in found:
                continue
            recent = grp.tail(4)
            last = recent.iloc[-1]
            coverage = last["total"] / last["debt_lcu_bn"] if pd.notna(last["debt_lcu_bn"]) else float("nan")
            if pd.notna(coverage) and coverage < MIN_COVERAGE[sector]:
                continue  # only part of the debt is reported
            share = float(recent["share"].mean())
            if not PLAUSIBLE[0] <= share <= PLAUSIBLE[1]:
                continue
            found[iso3] = {
                "maturing_share": round(share, 1),
                "method": "qpsd_due_12m",
                "source": f"World Bank QPSD: {SECTORS[sector]} debt due within 12 months",
                "as_of": last["date"],
                "quality": "high",
            }
    return found


def _external_amortisation() -> tuple[dict, str]:
    """{iso3: (rate %, last year)} for external public debt, and which method was used."""
    stock = _wb_series("DT.DOD.DPPG.CD", source=2, dates="2019:2025")
    try:
        flow = _wb_series("DT.AMT.DPPG.CD", source=6, dates="2019:2025")  # principal repaid
        method = "principal"
    except (requests.RequestException, ValueError):
        flow = _wb_series("DT.TDS.DPPG.CD", source=2, dates="2019:2025")  # principal + interest
        method = "service_minus_interest"
    for s in (stock, flow):
        s["year"] = s["date"].astype(int)
    m = flow.merge(stock, on=["iso3", "year"], suffixes=("_flow", "_stock")).sort_values("year")
    m["stock_prev"] = m.groupby("iso3")["value_stock"].shift(1)
    m = m[m["stock_prev"] > 0]
    m["rate"] = 100 * m["value_flow"] / m["stock_prev"]
    if method == "service_minus_interest":
        m["rate"] = (m["rate"] - ASSUMED_EXTERNAL_INTEREST).clip(lower=1.0)
    # Average the last three years: repayments are lumpy.
    out = {}
    for iso3, grp in m.groupby("iso3"):
        recent = grp.tail(3)
        out[iso3] = (float(recent["rate"].mean()), int(recent["year"].max()))
    return out, method


def build_table() -> pd.DataFrame:
    from .weo import list_countries

    weo = _weo_debt_levels()
    direct = _qpsd_due_within_year(weo)

    try:
        external, ext_method = _external_amortisation()
    except (requests.RequestException, ValueError) as exc:
        print(f"Skipping external debt repayments (World Bank unavailable: {exc}).")
        external, ext_method = {}, ""

    fx = fx_share.load_table().set_index("iso3")["fx_share"] if fx_share.TABLE_PATH.exists() else pd.Series(dtype=float)

    # Typical domestic rollover rate. For emerging/developing countries where we
    # know both the total share due within a year and the external repayment
    # rate, back out the domestic part:
    #   total = a * external + (1 - a) * domestic  =>  domestic = (total - a * external) / (1 - a)
    implied = []
    for iso3, est in direct.items():
        if iso3 in ADVANCED or iso3 not in external or pd.isna(fx.get(iso3)):
            continue
        a = float(fx[iso3]) / 100
        if a > 0.8:  # almost no domestic debt: the back-out is unreliable
            continue
        implied.append((est["maturing_share"] - a * external[iso3][0]) / (1 - a))
    implied = [x for x in implied if 0 < x < 100]
    domestic_typical = float(pd.Series(implied).median()) if len(implied) >= 5 else DEFAULT_SHARE
    print(f"Typical domestic rollover rate: {domestic_typical:.1f}% (from {len(implied)} countries)")
    rows = []
    # Same country list as the foreign-currency table (drops regional aggregates).
    countries = sorted(fx.index) if len(fx) else sorted(list_countries())
    for iso3 in countries:
        if iso3 in direct:
            est = direct[iso3]
        elif iso3 in external and pd.notna(fx.get(iso3)):
            rate, year = external[iso3]
            a = float(fx[iso3]) / 100
            share = a * rate + (1 - a) * domestic_typical
            how = "principal repayments" if ext_method == "principal" else f"debt service minus {ASSUMED_EXTERNAL_INTEREST:.0f}% assumed interest"
            est = {
                "maturing_share": round(share, 1),
                "method": f"external_{ext_method}",
                "source": (
                    f"External debt: {rate:.1f}% repaid a year (World Bank, {how}); "
                    f"domestic debt: {domestic_typical:.0f}% (typical for emerging/developing countries); "
                    f"weighted by the {a * 100:.0f}% foreign-currency share"
                ),
                "as_of": str(year),
                "quality": "medium" if ext_method == "principal" else "low",
            }
        else:
            est = {
                "maturing_share": DEFAULT_SHARE,
                "method": "default",
                "source": "No country data: default of 25% (about a 4-year average maturity)",
                "as_of": "",
                "quality": "none",
            }
        rows.append({"iso3": iso3, **est})
    return pd.DataFrame(rows)


def lookup(iso3: str) -> dict:
    default = {"iso3": iso3, "maturing_share": DEFAULT_SHARE, "method": "default", "source": "Default of 25%", "as_of": "", "quality": "none"}
    if not TABLE_PATH.exists():
        return default
    t = pd.read_csv(TABLE_PATH, dtype={"as_of": str}).fillna({"as_of": ""})
    row = t[t["iso3"] == iso3.upper()]
    return row.iloc[0].to_dict() if not row.empty else default


if __name__ == "__main__":
    table = build_table()
    table.to_csv(TABLE_PATH, index=False)
    print(table["method"].value_counts().to_string())
    print(f"Saved {len(table)} countries to {TABLE_PATH}")
