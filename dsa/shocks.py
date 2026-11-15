"""Shocks and stress tests.

In plain English: each function takes the baseline assumptions and returns a
shocked copy. The debt engine then re-runs on the shocked copy. Where the IMF
papers specify knock-on effects between variables, we apply them:

* LIC-DSF Guidance Note (2018), Table 8: standardised stress tests and their
  interactions (growth shocks lower inflation, depreciation passes through to
  inflation, a weaker primary balance raises borrowing costs).
* MAC SRDSF review (2021), Table 5: commodity price shock (fiscal impact of
  oil price moves) and natural disaster shock.

Shocked years are given as positions in the projection: year 1 is the first
projection year. The IMF's standard tests hit years 2 and 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import Assumptions

# Interaction elasticities, LIC-DSF Guidance Note 2018, Table 8.
INFLATION_PER_GROWTH = 0.6  # deflator inflation falls 0.6pp per 1pp lower growth
FX_PASS_THROUGH = 0.3  # inflation rises 0.3pp per 1% depreciation, in the shock year
RATE_PER_PB_WORSENING = 0.25  # +25bp borrowing cost per 1pp of GDP weaker primary balance

# Commodity shock fiscal impact, MAC SRDSF 2021, Table 5 (Spatafora and Samake, 2012).
EXPORTER_REVENUE_PER_10PCT = 0.5  # revenue falls 0.5pp of GDP per 10% fall in oil price
IMPORTER_SPENDING_PER_10PCT = 0.9  # spending rises 0.9pp of GDP per 10% rise in oil price


def _years(a: Assumptions, positions: list[int]) -> list[int]:
    ys = a.years
    return [ys[p - 1] for p in positions if 1 <= p <= len(ys)]


@dataclass
class CustomShock:
    """The sliders in the app. Zero means no shock."""

    growth_pp: float = 0.0  # change in real growth, pp (negative = slower)
    growth_years: tuple[int, int] = (2, 3)
    interest_bp: float = 0.0  # change in market rate on new borrowing, basis points
    interest_years: tuple[int, int] = (2, 10)
    primary_balance_pp: float = 0.0  # change in primary balance, pp of GDP (negative = worse)
    pb_years: tuple[int, int] = (2, 3)
    depreciation_pct: float = 0.0  # one-off depreciation, %
    depreciation_year: int = 2
    oil_price_pct: float = 0.0  # change in oil price, % (negative = fall), sustained
    oil_years: tuple[int, int] = (2, 3)
    oil_role: str = "none"  # "exporter", "importer" or "none"
    oil_fiscal_pp_per_10pct: float | None = None  # override the IMF default sensitivity
    oil_growth_pp_per_10pct: float = 0.0  # knock-on to growth (paper gives no default)
    contingent_liability_pp: float = 0.0  # one-off jump in debt, % of GDP
    contingent_year: int = 2
    interactions: bool = True  # apply the IMF knock-on effects


def _span(a: Assumptions, span: tuple[int, int]) -> list[int]:
    lo, hi = span
    return _years(a, list(range(lo, hi + 1)))


def apply_growth_shock(a: Assumptions, delta: pd.Series, interactions: bool = True) -> None:
    """Change real growth by `delta` pp in the given years, in place.

    With interactions (LIC-DSF Table 8, B1): inflation moves 0.6pp per 1pp of
    growth, and spending stays at its baseline level in money terms, so
    spending / GDP rises when GDP is smaller, weakening the primary balance.
    The spending effect lasts for the shock years only; afterwards spending
    adjusts back to its baseline share of GDP (our reading of the template).
    """
    ratio = 1.0  # shocked nominal GDP / baseline nominal GDP
    for y in a.years:
        if y not in delta.index:
            continue
        d = float(delta[y])
        dpi = INFLATION_PER_GROWTH * d if interactions else 0.0
        g0, p0 = a.real_growth[y], a.deflator_growth[y]
        a.real_growth[y] = g0 + d
        a.deflator_growth[y] = p0 + dpi
        ratio *= (1 + (g0 + d) / 100) * (1 + (p0 + dpi) / 100) / ((1 + g0 / 100) * (1 + p0 / 100))
        if interactions and a.expenditure is not None:
            a.primary_balance[y] -= a.expenditure[y] * (1 / ratio - 1)


def apply_interest_shock(a: Assumptions, bp: float, years: list[int]) -> None:
    """Higher market rates feed into the effective rate as debt is refinanced."""
    m = a.repricing_share / 100
    for k, y in enumerate(a.years):
        if y < years[0]:
            continue
        repriced = min(1.0, m * (k - a.years.index(years[0]) + 1))
        a.effective_rate[y] += bp / 100 * repriced


def custom(a: Assumptions, s: CustomShock) -> Assumptions:
    out = a.copy()
    if s.growth_pp:
        ys = _span(out, s.growth_years)
        apply_growth_shock(out, pd.Series(s.growth_pp, index=ys), s.interactions)

    if s.primary_balance_pp:
        for y in _span(out, s.pb_years):
            out.primary_balance[y] += s.primary_balance_pp
        if s.interactions and s.primary_balance_pp < 0:
            ys = _span(out, s.pb_years)
            apply_interest_shock(out, -s.primary_balance_pp * RATE_PER_PB_WORSENING * 100, ys)

    if s.interest_bp:
        apply_interest_shock(out, s.interest_bp, _span(out, s.interest_years))

    if s.depreciation_pct:
        ys = _years(out, [s.depreciation_year])
        for y in ys:
            out.fx_depreciation[y] += s.depreciation_pct
            if s.interactions:
                out.deflator_growth[y] += FX_PASS_THROUGH * s.depreciation_pct

    if s.oil_price_pct and s.oil_role in ("exporter", "importer"):
        ys = _span(out, s.oil_years)
        tens = s.oil_price_pct / 10
        if s.oil_role == "exporter":
            per10 = EXPORTER_REVENUE_PER_10PCT if s.oil_fiscal_pp_per_10pct is None else s.oil_fiscal_pp_per_10pct
            pb_change = per10 * tens  # oil falls (tens < 0) -> revenue and primary balance fall
        else:
            per10 = IMPORTER_SPENDING_PER_10PCT if s.oil_fiscal_pp_per_10pct is None else s.oil_fiscal_pp_per_10pct
            pb_change = -per10 * tens  # oil rises (tens > 0) -> spending rises, balance falls
        for y in ys:
            out.primary_balance[y] += pb_change
            if out.oil_price is not None:
                out.oil_price[y] *= 1 + s.oil_price_pct / 100
        if s.oil_growth_pp_per_10pct:
            sign = 1 if s.oil_role == "exporter" else -1
            apply_growth_shock(out, pd.Series(sign * s.oil_growth_pp_per_10pct * tens, index=ys), s.interactions)

    if s.contingent_liability_pp:
        for y in _years(out, [s.contingent_year]):
            out.other_flows[y] += s.contingent_liability_pp
    return out


# ---------------------------------------------------------------- IMF presets

def _one_sd_shock(a: Assumptions, col: str, stats: pd.DataFrame, years: list[int]) -> pd.Series:
    """LIC-DSF rule: shocked value = min(baseline, historical average) - 1 SD."""
    base = getattr(a, col)
    mean, sd = stats.loc[col, "mean"], stats.loc[col, "std"]
    return pd.Series({y: min(base[y], mean) - sd - base[y] for y in years})


def historical_scenario(a: Assumptions, stats: pd.DataFrame) -> Assumptions:
    """A1: growth, primary balance, deflator and interest rate at their 10-year averages."""
    out = a.copy()
    for col in ("real_growth", "primary_balance", "deflator_growth", "effective_rate"):
        getattr(out, col)[:] = stats.loc[col, "mean"]
    out.fx_depreciation[:] = stats.loc["fx_depreciation", "mean"]
    out.other_flows[:] = 0.0
    return out


def preset(name: str, a: Assumptions, stats: pd.DataFrame, half: bool = False) -> Assumptions:
    """IMF standard stress tests. `half` applies the shock at half size (combined test)."""
    k = 0.5 if half else 1.0
    ys = _years(a, [2, 3])
    out = a.copy()
    if name == "growth":  # LIC-DSF B1
        apply_growth_shock(out, _one_sd_shock(a, "real_growth", stats, ys) * k)
    elif name == "primary_balance":  # LIC-DSF B2
        delta = _one_sd_shock(a, "primary_balance", stats, ys) * k
        for y, d in delta.items():
            out.primary_balance[y] += d
        worst = -float(delta.min())
        if worst > 0:
            apply_interest_shock(out, worst * RATE_PER_PB_WORSENING * 100, ys)
    elif name == "depreciation":  # LIC-DSF B5: 30% one-off in year 2
        y = ys[0]
        out.fx_depreciation[y] += 30 * k
        out.deflator_growth[y] += FX_PASS_THROUGH * 30 * k
    elif name == "interest":  # +200bp on market rates, passed through as debt reprices
        apply_interest_shock(out, 200 * k, ys)
    elif name == "contingent":  # LIC-DSF contingent liability minimum: +5% of GDP in year 2
        out.other_flows[ys[0]] += 5 * k
    elif name == "natural_disaster":  # MAC SRDSF Table 5: +7pp debt, -5pp growth
        out.other_flows[ys[0]] += 7 * k
        apply_growth_shock(out, pd.Series(-5 * k, index=[ys[0]]))
    elif name == "combined":  # LIC-DSF B6: B1, B2 and B5 at half size
        for part in ("growth", "primary_balance", "depreciation"):
            shocked = preset(part, a, stats, half=True)
            for col in ("real_growth", "deflator_growth", "primary_balance", "effective_rate", "fx_depreciation"):
                getattr(out, col)[:] += getattr(shocked, col) - getattr(a, col)
    else:
        raise ValueError(f"Unknown stress test '{name}'")
    return out


PRESETS = {
    "growth": "Real GDP growth: 1 SD below (LIC-DSF B1)",
    "primary_balance": "Primary balance: 1 SD below (LIC-DSF B2)",
    "depreciation": "30% depreciation in year 2 (LIC-DSF B5)",
    "interest": "Interest rates +200bp",
    "contingent": "Contingent liability: +5% of GDP (LIC-DSF)",
    "natural_disaster": "Natural disaster: +7% of GDP debt, -5pp growth (MAC SRDSF)",
    "combined": "Combined: growth, primary balance, depreciation at half size (LIC-DSF B6)",
}
