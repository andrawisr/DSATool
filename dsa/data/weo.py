"""Download DSA inputs from the IMF World Economic Outlook (WEO).

In plain English: this file asks the IMF's free public data service for the
WEO series a debt sustainability analysis needs, history plus projections,
for one country. It saves a copy on your laptop so the next request is
instant, and turns the raw series into the variables the DSA engine uses
(effective interest rate, exchange rate, and so on).

Source: IMF SDMX API, dataflow IMF.RES/WEO (the current WEO release).
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

SDMX_URL = "https://api.imf.org/external/sdmx/2.1/data/IMF.RES,WEO/{countries}.{indicators}.A"
COUNTRY_NAMES_URL = "https://www.imf.org/external/datamapper/api/v1/countries"
WORLD = "G001"  # WEO code for the world aggregate, where the oil price lives

# WEO series code -> what we call it, and a plain-English description.
COUNTRY_SERIES = {
    "GGXWDG_NGDP": ("debt", "General government gross debt, % of GDP"),
    "GGXONLB_NGDP": ("primary_balance", "Primary balance (net lending excl. interest), % of GDP"),
    "GGXCNL_NGDP": ("overall_balance", "Overall balance (net lending/borrowing), % of GDP"),
    "GGR_NGDP": ("revenue", "General government revenue, % of GDP"),
    "GGX_NGDP": ("expenditure", "General government total expenditure, % of GDP"),
    "NGDP_RPCH": ("real_growth", "Real GDP growth, %"),
    "NGDP_D": ("deflator_index", "GDP deflator, index"),
    "PCPIPCH": ("cpi_inflation", "CPI inflation (average), %"),
    "NGDP": ("ngdp_lcu", "Nominal GDP, local currency"),
    "NGDPD": ("ngdp_usd", "Nominal GDP, US$"),
    "BCA_NGDPD": ("current_account", "Current account balance, % of GDP"),
}
WORLD_SERIES = {
    "POILAPSP": ("oil_price", "Crude oil price, simple average of Brent/Dubai/WTI, US$ per barrel"),
}

# Descriptions of the variables we work out ourselves, shown in the app.
DERIVED_SERIES = {
    "interest": "Interest paid, % of GDP (primary balance minus overall balance)",
    "effective_rate": "Effective nominal interest rate, % (interest / previous year's debt stock)",
    "deflator_growth": "GDP deflator inflation, %",
    "nominal_growth": "Nominal GDP growth, %",
    "fx_lcu_per_usd": "Implied exchange rate, local currency per US$ (nominal GDP LCU / US$)",
    "fx_depreciation": "Depreciation vs US$, % (positive = local currency weaker)",
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
DEFAULT_MAX_AGE_DAYS = 7


@dataclass
class CountryData:
    """Everything we know about one country, ready for the DSA engine."""

    iso3: str
    name: str
    table: pd.DataFrame  # rows = years, columns = variables
    last_actual_year: int | None  # last year of outturn data; later years are WEO projections
    downloaded: str  # date the data was fetched from the IMF

    @property
    def projection_years(self) -> list[int]:
        if self.last_actual_year is None:
            return []
        return [y for y in self.table.index if y > self.last_actual_year]


# ---------------------------------------------------------------- download

def _fetch_sdmx(countries: list[str], codes: list[str], start: int, timeout: int = 90) -> pd.DataFrame:
    url = SDMX_URL.format(countries="+".join(countries), indicators="+".join(codes))
    resp = requests.get(
        url,
        params={"startPeriod": start},
        headers={"Accept": "application/vnd.sdmx.data+csv"},
        timeout=timeout,
    )
    if resp.status_code == 404:
        return pd.DataFrame()  # the IMF returns 404 when no data matches
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    keep = ["COUNTRY", "INDICATOR", "TIME_PERIOD", "OBS_VALUE", "LATEST_ACTUAL_ANNUAL_DATA"]
    return raw[[c for c in keep if c in raw.columns]]


def _build_table(raw: pd.DataFrame, iso3: str) -> tuple[pd.DataFrame, int | None]:
    names = {code: name for code, (name, _) in {**COUNTRY_SERIES, **WORLD_SERIES}.items()}
    raw = raw.copy()
    raw["TIME_PERIOD"] = pd.to_numeric(raw["TIME_PERIOD"], errors="coerce").astype("Int64")
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
    raw = raw.dropna(subset=["TIME_PERIOD"])

    table = (
        raw.pivot_table(index="TIME_PERIOD", columns="INDICATOR", values="OBS_VALUE", aggfunc="first")
        .rename(columns=names)
        .sort_index()
    )
    table.index = table.index.astype(int)
    table.index.name = "year"
    for name in names.values():
        if name not in table.columns:
            table[name] = float("nan")

    last_actual = None
    country_rows = raw[raw["COUNTRY"] == iso3]
    if "LATEST_ACTUAL_ANNUAL_DATA" in country_rows:
        # Use the year the debt series ends its outturn data; fall back to growth.
        for code in ("GGXWDG_NGDP", "NGDP_RPCH"):
            vals = country_rows.loc[country_rows["INDICATOR"] == code, "LATEST_ACTUAL_ANNUAL_DATA"].dropna()
            if not vals.empty:
                last_actual = int(vals.iloc[0])
                break

    return add_derived_series(table), last_actual


def add_derived_series(t: pd.DataFrame) -> pd.DataFrame:
    """Work out the DSA variables the WEO doesn't publish directly."""
    t = t.copy()
    # Interest bill = primary balance - overall balance (both % of GDP).
    t["interest"] = t["primary_balance"] - t["overall_balance"]
    # Nominal GDP growth and deflator inflation.
    t["nominal_growth"] = t["ngdp_lcu"].pct_change(fill_method=None) * 100
    t["deflator_growth"] = t["deflator_index"].pct_change(fill_method=None) * 100
    # Effective interest rate: this year's interest bill / last year's debt stock,
    # both in local currency (IMF definition, Table A2 note 4 of the 2021 review).
    debt_lcu = t["debt"] / 100 * t["ngdp_lcu"]
    interest_lcu = t["interest"] / 100 * t["ngdp_lcu"]
    t["effective_rate"] = interest_lcu / debt_lcu.shift(1) * 100
    # Exchange rate implied by GDP in local currency vs GDP in dollars (annual average).
    t["fx_lcu_per_usd"] = t["ngdp_lcu"] / t["ngdp_usd"]
    t["fx_depreciation"] = t["fx_lcu_per_usd"].pct_change(fill_method=None) * 100
    return t


# ---------------------------------------------------------------- cache

def _cache_path(cache_dir: Path, iso3: str) -> Path:
    return cache_dir / f"weo_{iso3}.csv"


def _meta_path(cache_dir: Path, iso3: str) -> Path:
    return cache_dir / f"weo_{iso3}.json"


def _read_cache(cache_dir: Path, iso3: str, max_age_days: int, start: int) -> CountryData | None:
    csv, meta = _cache_path(cache_dir, iso3), _meta_path(cache_dir, iso3)
    if not (csv.exists() and meta.exists()):
        return None
    info = json.loads(meta.read_text())
    age = (date.today() - date.fromisoformat(info["downloaded"])).days
    if age > max_age_days or info.get("start", 9999) > start:
        return None  # too old, or doesn't go back far enough
    table = pd.read_csv(csv, index_col="year")
    return CountryData(iso3, info["name"], table, info["last_actual_year"], info["downloaded"])


def _write_cache(cache_dir: Path, data: CountryData, start: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data.table.to_csv(_cache_path(cache_dir, data.iso3))
    _meta_path(cache_dir, data.iso3).write_text(
        json.dumps(
            {"name": data.name, "last_actual_year": data.last_actual_year, "downloaded": data.downloaded, "start": start},
            indent=2,
        )
    )


# ---------------------------------------------------------------- public API

def list_countries(cache_dir: Path = DEFAULT_CACHE_DIR, timeout: int = 30) -> dict[str, str]:
    """Return {ISO3 code: country name} for the countries the IMF covers."""
    path = cache_dir / "countries.json"
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get(COUNTRY_NAMES_URL, timeout=timeout)
    resp.raise_for_status()
    names = {k: v["label"] for k, v in resp.json()["countries"].items() if v and v.get("label")}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(names, indent=1, sort_keys=True))
    return names


def get_country(
    iso3: str,
    start: int = 1990,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    refresh: bool = False,
) -> CountryData:
    """Get WEO history and projections for one country (from the cache if it's fresh).

    `iso3` is the three-letter IMF code, e.g. "BRA" for Brazil.
    """
    iso3 = iso3.upper()
    if not refresh:
        cached = _read_cache(cache_dir, iso3, max_age_days, start)
        if cached is not None:
            return cached

    country_raw = _fetch_sdmx([iso3], list(COUNTRY_SERIES), start)
    if country_raw.empty:
        raise ValueError(f"The IMF WEO has no data for country code '{iso3}'.")
    world_raw = _fetch_sdmx([WORLD], list(WORLD_SERIES), start)
    table, last_actual = _build_table(pd.concat([country_raw, world_raw]), iso3)

    try:
        name = list_countries(cache_dir).get(iso3, iso3)
    except requests.RequestException:
        name = iso3
    data = CountryData(iso3, name, table, last_actual, datetime.now().date().isoformat())
    _write_cache(cache_dir, data, start)
    return data
