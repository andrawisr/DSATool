"""Estimate each country's net fuel exports as a share of GDP.

In plain English: when the oil price moves, a country's current account moves
by roughly (net fuel exports / GDP) x (price change). An exporter's current
account improves when oil rises; an importer's worsens. This file works out
net fuel exports for every country from World Bank trade data:

    net fuel exports = fuel exports - fuel imports
      fuel exports = fuel share of merchandise exports x merchandise exports
      fuel imports = fuel share of merchandise imports x merchandise imports

divided by GDP in US$ (from the IMF WEO), averaged over the last three years
available. "Fuel" includes oil, gas and coal, so for gas exporters the
sensitivity to the oil price is an approximation.

Run `python -m dsa.data.oil_trade` to rebuild the table. A figure in
country_overrides.csv (field `net_oil_exports`) takes priority.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fx_share import _wb_series

TABLE_PATH = Path(__file__).with_name("oil_trade_estimates.csv")


def build_table() -> pd.DataFrame:
    parts = {}
    for key, ind in {
        "fuel_x_share": "TX.VAL.FUEL.ZS.UN",
        "fuel_m_share": "TM.VAL.FUEL.ZS.UN",
        "merch_x": "TX.VAL.MRCH.CD.WT",
        "merch_m": "TM.VAL.MRCH.CD.WT",
    }.items():
        s = _wb_series(ind, source=2, dates="2018:2025")
        s["year"] = s["date"].astype(int)
        parts[key] = s.set_index(["iso3", "year"])["value"]
    t = pd.DataFrame(parts).dropna()

    gdp = _weo_gdp_usd()
    t = t.join(gdp, how="inner")
    t["net_fuel_exports"] = 100 * (t["fuel_x_share"] / 100 * t["merch_x"] - t["fuel_m_share"] / 100 * t["merch_m"]) / t["gdp_usd"]

    rows = []
    for iso3, grp in t.reset_index().sort_values("year").groupby("iso3"):
        recent = grp.tail(3)
        rows.append(
            {
                "iso3": iso3,
                "net_oil_exports": round(float(recent["net_fuel_exports"].mean()), 1),
                "fuel_exports": round(float((recent["fuel_x_share"] / 100 * recent["merch_x"] / recent["gdp_usd"] * 100).mean()), 1),
                "fuel_imports": round(float((recent["fuel_m_share"] / 100 * recent["merch_m"] / recent["gdp_usd"] * 100).mean()), 1),
                "as_of": f"{int(recent['year'].min())}-{int(recent['year'].max())}",
                "source": "World Bank WDI fuel trade / IMF WEO GDP in US$, 3-year average",
                "quality": "medium",
            }
        )
    return pd.DataFrame(rows)


def _weo_gdp_usd() -> pd.Series:
    import io

    import requests

    from .fx_share import WEO_URL

    resp = requests.get(
        WEO_URL.format(inds="NGDPD"),
        params={"startPeriod": 2018},
        headers={"Accept": "application/vnd.sdmx.data+csv"},
        timeout=180,
    )
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    raw = raw.rename(columns={"COUNTRY": "iso3", "TIME_PERIOD": "year", "OBS_VALUE": "gdp_usd"})
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    return raw.dropna(subset=["year"]).astype({"year": int}).set_index(["iso3", "year"])["gdp_usd"]


def lookup(iso3: str) -> dict:
    """Net fuel exports, % of GDP (negative = net importer). Overrides win."""
    from . import overrides

    o = overrides.get(iso3, "net_oil_exports")
    if o is not None:
        return {"iso3": iso3, "net_oil_exports": o["value"], "source": o["source"], "as_of": o["as_of"], "quality": o["quality"], "method": "override"}
    missing = {"iso3": iso3, "net_oil_exports": None, "source": "No fuel trade data", "as_of": "", "quality": "none", "method": "missing"}
    if not TABLE_PATH.exists():
        return missing
    t = pd.read_csv(TABLE_PATH, dtype={"as_of": str})
    row = t[t["iso3"] == iso3.upper()]
    if row.empty:
        return missing
    rec = row.iloc[0].to_dict()
    rec["method"] = "estimate"
    return rec


if __name__ == "__main__":
    table = build_table()
    table.to_csv(TABLE_PATH, index=False)
    print(f"Saved {len(table)} countries to {TABLE_PATH}")
