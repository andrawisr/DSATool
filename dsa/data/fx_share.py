"""Estimate the share of each country's government debt in foreign currency.

In plain English: the DSA needs to know how much debt is in foreign currency,
because a depreciation makes that debt bigger in local-currency terms. No single
free source covers every country, so we try sources in order of quality and
record which one we used:

  1. World Bank Quarterly Public Sector Debt (QPSD): the debt that governments
     report, split into foreign- and domestic-currency debt. This is the best
     source. We use general government, then central government, then
     budgetary central government, whichever is reported.
  2. QPSD split by creditor: the share owed to non-residents. This is a proxy,
     since external debt is mostly, but not always, in foreign currency.
  3. World Bank International Debt Statistics (IDS): external public and
     publicly guaranteed debt in US$, divided by WEO gross debt in US$. Covers
     low- and middle-income countries.
  4. Rules of thumb: euro area members and advanced economies that borrow in
     their own currency are set to ~0%.

Every estimate passes a coverage check: if the debt a country reports to the
World Bank is much smaller than its WEO gross debt (i.e. it only reported part
of its debt), the estimate is rejected and we fall back to the next source.

Run `python -m dsa.data.fx_share` to rebuild the table from fresh data.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

TABLE_PATH = Path(__file__).with_name("fx_share_estimates.csv")

WB_URL = "https://api.worldbank.org/v2/country/all/indicator/{ind}"
WEO_URL = "https://api.imf.org/external/sdmx/2.1/data/IMF.RES,WEO/.{inds}.A"

SECTORS = {"GG": "general government", "CG": "central government", "BC": "budgetary central government"}
MIN_COVERAGE = {"GG": 0.75, "CG": 0.6, "BC": 0.5}  # reported debt / WEO gross debt
OLDEST_QUARTER = "2019Q1"

EURO_AREA = {
    "AUT", "BEL", "HRV", "CYP", "EST", "FIN", "FRA", "DEU", "GRC", "IRL", "ITA",
    "LVA", "LTU", "LUX", "MLT", "NLD", "PRT", "SVK", "SVN", "ESP", "BGR",
}
# Advanced economies that borrow almost entirely in their own currency.
OWN_CURRENCY_ADVANCED = {
    "USA", "GBR", "JPN", "CAN", "AUS", "NZL", "CHE", "NOR", "SWE", "DNK",
    "KOR", "SGP", "HKG", "TWN", "ISR", "CZE",
    # Large economies with negligible foreign-currency government debt.
    "CHN", "IND",
}

QUALITY = {
    "qpsd_currency": "high",
    "qpsd_external": "medium",
    "ids_external": "medium",
    "rule_euro": "high",
    "rule_advanced": "medium",
    "missing": "none",
}


# ---------------------------------------------------------------- downloads

RAW_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "worldbank"


def _wb_series(indicator: str, source: int, dates: str, retries: int = 4) -> pd.DataFrame:
    """One World Bank series for all countries. Raw downloads are kept in
    data_cache/worldbank so an interrupted rebuild doesn't start from scratch;
    delete that folder to force fresh downloads."""
    raw_path = RAW_DIR / f"{indicator}.json"
    if raw_path.exists():
        payload = json.loads(raw_path.read_text())
    else:
        params = {"source": source, "format": "json", "per_page": 30000, "date": dates}
        for attempt in range(retries):
            try:
                # (connect timeout, read timeout): give up on a stalled transfer and retry.
                resp = requests.get(WB_URL.format(ind=indicator), params=params, timeout=(15, 60))
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or len(payload) < 2:
                    raise ValueError(f"World Bank returned an error for {indicator}: {payload}")
                break
            except (requests.RequestException, ValueError):
                if attempt == retries - 1:
                    raise
                time.sleep(3)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload))
    rows = payload[1] if len(payload) > 1 and payload[1] else []
    out = pd.DataFrame(
        [(r["countryiso3code"], r["date"], r["value"]) for r in rows if r["value"] is not None],
        columns=["iso3", "date", "value"],
    )
    return out[out["iso3"].str.len() == 3]


def _weo_debt_levels(start: int = 2018) -> pd.DataFrame:
    """WEO gross debt in local currency and US$ (full units, as the IMF API returns them), all countries."""
    resp = requests.get(
        WEO_URL.format(inds="GGXWDG_NGDP+NGDP+NGDPD"),
        params={"startPeriod": start},
        headers={"Accept": "application/vnd.sdmx.data+csv"},
        timeout=180,
    )
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    t = raw.pivot_table(index=["COUNTRY", "TIME_PERIOD"], columns="INDICATOR", values="OBS_VALUE").reset_index()
    t.columns.name = None
    t = t.rename(columns={"COUNTRY": "iso3", "TIME_PERIOD": "year"})
    t["debt_lcu_bn"] = t["GGXWDG_NGDP"] / 100 * t["NGDP"]
    t["debt_usd_bn"] = t["GGXWDG_NGDP"] / 100 * t["NGDPD"]
    return t[["iso3", "year", "debt_lcu_bn", "debt_usd_bn"]]


# ---------------------------------------------------------------- estimation

def _qpsd_candidates(weo: pd.DataFrame) -> tuple[dict, dict]:
    """Return {iso3: estimate} from the QPSD currency split and from the creditor split."""
    frames = []
    for sector in SECTORS:
        for kind in ("DECF", "DECN", "DECX", "DECT"):
            s = _wb_series(f"DP.DOD.{kind}.CR.{sector}", source=20, dates=f"{OLDEST_QUARTER}:2030Q4")
            frames.append(s.assign(sector=sector, kind=kind))
    q = pd.concat(frames)
    wide = q.pivot_table(index=["iso3", "sector", "date"], columns="kind", values="value").reset_index()
    wide.columns.name = None
    wide["year"] = wide["date"].str[:4].astype(int)
    wide = wide.merge(weo, on=["iso3", "year"], how="left")

    def pick(frame, split_cols, share_fn, method):
        found = {}
        for sector in SECTORS:  # GG first, then CG, then BC
            rows = frame[(frame["sector"] == sector)].dropna(subset=split_cols)
            for iso3, grp in rows.sort_values("date").groupby("iso3"):
                if iso3 in found:
                    continue
                last = grp.iloc[-1]
                reported_bn = last["reported"] if pd.notna(last.get("reported")) else float("nan")
                coverage = reported_bn / last["debt_lcu_bn"] if pd.notna(last["debt_lcu_bn"]) else float("nan")
                if pd.notna(coverage) and coverage < MIN_COVERAGE[sector]:
                    continue  # the country reported only part of its debt
                found[iso3] = {
                    "fx_share": round(float(share_fn(last)), 1),
                    "method": method,
                    "source": f"World Bank QPSD, {SECTORS[sector]}",
                    "as_of": last["date"],
                    "coverage_vs_weo": round(float(coverage), 2) if pd.notna(coverage) else None,
                }
        return found

    by_currency = pick(
        wide.assign(reported=wide["DECF"] + wide["DECN"]),
        ["DECF", "DECN"],
        lambda r: 100 * r["DECF"] / (r["DECF"] + r["DECN"]),
        "qpsd_currency",
    )
    by_creditor = pick(
        wide.assign(reported=wide["DECT"]),
        ["DECX", "DECT"],
        lambda r: 100 * r["DECX"] / r["DECT"],
        "qpsd_external",
    )
    return by_currency, by_creditor


def _ids_candidates(weo: pd.DataFrame) -> dict:
    try:
        ids = _wb_series("DT.DOD.DPPG.CD", source=6, dates="2019:2025")
    except (requests.RequestException, ValueError) as exc:
        print(f"Skipping World Bank IDS (service unavailable: {exc}). Re-run later to fill gaps.")
        return {}
    ids["year"] = ids["date"].astype(int)
    m = ids.merge(weo, on=["iso3", "year"], how="inner").dropna(subset=["debt_usd_bn"])
    found = {}
    for iso3, grp in m.sort_values("year").groupby("iso3"):
        last = grp.iloc[-1]
        share = 100 * last["value"] / last["debt_usd_bn"]
        if share > 105:  # external PPG > total gross debt: definitions don't line up
            continue
        found[iso3] = {
            "fx_share": round(min(float(share), 100.0), 1),
            "method": "ids_external",
            "source": "World Bank IDS external public & publicly guaranteed debt / WEO gross debt",
            "as_of": str(int(last["year"])),
            "coverage_vs_weo": None,
        }
    return found


def build_table() -> pd.DataFrame:
    weo = _weo_debt_levels()
    by_currency, by_creditor = _qpsd_candidates(weo)
    by_ids = _ids_candidates(weo)
    countries = sorted(set(weo["iso3"]) | set(by_currency) | set(by_creditor) | set(by_ids))
    try:  # keep only real countries (drop regional aggregates and duplicate codes)
        from .weo import list_countries

        known = set(list_countries())
        countries = [c for c in countries if c in known]
    except requests.RequestException:
        pass

    rows = []
    for iso3 in countries:
        if iso3.startswith("G") and iso3[1:].isdigit():
            continue  # WEO regional aggregates, not countries
        if iso3 in EURO_AREA:
            est = {"fx_share": 0.0, "method": "rule_euro", "source": "Euro area member: euro is the domestic currency", "as_of": "", "coverage_vs_weo": None}
            if iso3 in by_currency:  # keep the small non-euro share where it is reported
                est = by_currency[iso3]
        elif iso3 in by_currency:
            est = by_currency[iso3]
        elif iso3 in OWN_CURRENCY_ADVANCED:
            est = {"fx_share": 1.0, "method": "rule_advanced", "source": "Advanced economy borrowing in own currency (assumed)", "as_of": "", "coverage_vs_weo": None}
        elif iso3 in by_creditor:
            est = by_creditor[iso3]
        elif iso3 in by_ids:
            est = by_ids[iso3]
        else:
            est = {"fx_share": None, "method": "missing", "source": "No estimate found: enter from the Article IV", "as_of": "", "coverage_vs_weo": None}
        rows.append({"iso3": iso3, **est, "quality": QUALITY[est["method"]]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- lookup

def load_table() -> pd.DataFrame:
    return pd.read_csv(TABLE_PATH, dtype={"as_of": str})


def lookup(iso3: str) -> dict:
    """The estimate for one country, as a dict (fx_share is None when unknown)."""
    if not TABLE_PATH.exists():
        return {"iso3": iso3, "fx_share": None, "method": "missing", "source": "Estimates table not built yet", "as_of": "", "quality": "none"}
    t = load_table()
    row = t[t["iso3"] == iso3.upper()]
    if row.empty:
        return {"iso3": iso3, "fx_share": None, "method": "missing", "source": "Not in table", "as_of": "", "quality": "none"}
    rec = row.iloc[0].to_dict()
    if pd.isna(rec["fx_share"]):
        rec["fx_share"] = None
    return rec


if __name__ == "__main__":
    table = build_table()
    table.to_csv(TABLE_PATH, index=False)
    print(table["method"].value_counts().to_string())
    print(f"Saved {len(table)} countries to {TABLE_PATH}")
