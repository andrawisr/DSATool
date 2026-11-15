"""The DSA engine: public debt dynamics, IMF style.

In plain English: given last year's debt and this year's interest rate,
growth, inflation, exchange rate move and primary balance, work out this
year's debt, and break the change into its drivers exactly as the IMF does
(note 5 of Table A2 in the 2021 MAC review):

    automatic debt dynamics = [r - pi(1+g) - g + a*e(1+r)] / (1 + g + pi + g*pi) * debt(t-1)

      r  = effective nominal interest rate      pi = GDP deflator inflation
      g  = real GDP growth                      a  = share of debt in foreign currency
      e  = depreciation of the local currency vs US$

    change in debt = automatic debt dynamics - primary balance + other flows

Rates are in percent per year (5 means 5%), shares and debt in percent too.
The engine knows nothing about data sources or screens; it only does the maths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

DRIVERS = ["effective_rate", "real_growth", "deflator_growth", "fx_depreciation", "primary_balance", "other_flows"]


@dataclass
class Assumptions:
    """Inputs for one scenario. Every series is indexed by projection year."""

    initial_debt: float  # debt / GDP (%) at the end of the last actual year
    effective_rate: pd.Series  # %
    real_growth: pd.Series  # %
    deflator_growth: pd.Series  # %
    fx_depreciation: pd.Series  # %, positive = local currency weaker
    primary_balance: pd.Series  # % of GDP, positive = surplus
    other_flows: pd.Series  # % of GDP: stock-flow adjustments, privatisation, etc.
    fx_share: float = 0.0  # % of debt in foreign currency
    repricing_share: float = 25.0  # % of the debt stock maturing / repricing each year
    expenditure: pd.Series | None = None  # primary expenditure, % of GDP (used by shocks)
    oil_price: pd.Series | None = None  # US$/bbl (used by shocks)
    current_account: pd.Series | None = None  # % of GDP (shown, and moved by balance-of-payments shocks)
    extra_maturing: pd.Series | None = None  # % of last year's debt falling due on top of repricing_share
    notes: list[str] = field(default_factory=list)

    @property
    def years(self) -> list[int]:
        return list(self.real_growth.index)

    def copy(self, **changes) -> "Assumptions":
        series = {k: getattr(self, k).copy() for k in DRIVERS}
        for opt in ("expenditure", "oil_price", "current_account", "extra_maturing"):
            if getattr(self, opt) is not None:
                series[opt] = getattr(self, opt).copy()
        return replace(self, **{**series, "notes": list(self.notes), **changes})


def debt_path(a: Assumptions) -> pd.DataFrame:
    """Project debt and its IMF-style decomposition, one row per year."""
    rows = []
    debt_prev = a.initial_debt
    alpha = a.fx_share / 100
    m = a.repricing_share / 100
    for y in a.years:
        r, g, pi, e = (s[y] / 100 for s in (a.effective_rate, a.real_growth, a.deflator_growth, a.fx_depreciation))
        pb, sfa = a.primary_balance[y], a.other_flows[y]
        parts = automatic_dynamics(debt_prev, r, g, pi, e, alpha)
        automatic = parts["real_interest_rate"] + parts["real_gdp_growth"] + parts["exchange_rate"]
        debt = debt_prev + automatic - pb + sfa
        denom = (1 + g) * (1 + pi)
        # Foreign-currency interest and repayments cost more in local currency
        # after a depreciation (the same revaluation as in the a*e*(1+r) term).
        fx_factor = 1 + alpha * e
        maturing = m + (a.extra_maturing[y] / 100 if a.extra_maturing is not None else 0.0)
        interest = r * debt_prev * fx_factor / denom  # interest bill, % of this year's GDP
        amortisation = maturing * debt_prev * fx_factor / denom  # share of debt falling due this year
        rows.append(
            {
                "year": y,
                "debt": debt,
                "change_in_debt": debt - debt_prev,
                "identified_flows": automatic - pb + sfa,
                "primary_deficit": -pb,
                "automatic_debt_dynamics": automatic,
                "interest_growth_differential": parts["real_interest_rate"] + parts["real_gdp_growth"],
                **parts,
                "other_flows": sfa,
                "interest_bill": interest,
                "amortisation": amortisation,
                "gross_financing_need": -pb + interest + amortisation,
                # Primary balance that would have kept debt / GDP flat this year.
                "debt_stabilising_pb": automatic + sfa,
            }
        )
        debt_prev = debt
    return pd.DataFrame(rows).set_index("year")


def automatic_dynamics(debt_prev: float, r: float, g: float, pi: float, e: float, alpha: float) -> dict:
    """IMF decomposition of the automatic debt dynamics (rates as fractions, debt in %)."""
    denom = 1 + g + pi + g * pi
    return {
        "real_interest_rate": (r - pi * (1 + g)) / denom * debt_prev,
        "real_gdp_growth": -g / denom * debt_prev,
        "exchange_rate": alpha * e * (1 + r) / denom * debt_prev,
    }


# ---------------------------------------------------------------- baseline from data

def baseline_from_weo(
    table: pd.DataFrame,
    last_actual_year: int,
    fx_share: float,
    horizon: int = 10,
    repricing_share: float = 25.0,
    overrides: dict[str, float] | None = None,
) -> Assumptions:
    """Turn WEO data into baseline assumptions.

    The WEO projects debt directly, and its path also reflects things the
    equation can't see (asset changes, valuation effects). As the IMF does, we
    call that gap the "residual" and carry it as `other_flows`, so the baseline
    reproduces the WEO debt path exactly. Shocks then move debt away from it.
    Beyond the last WEO year, drivers stay at their final value and the
    residual is set to zero (the IMF's convention for extending projections).
    """
    overrides = overrides or {}
    t = table.sort_index()
    weo_years = [y for y in t.index if y > last_actual_year and pd.notna(t.loc[y, "debt"])]
    years = list(range(last_actual_year + 1, last_actual_year + 1 + horizon))
    notes = []

    def series(col: str) -> pd.Series:
        s = t[col].reindex(years)
        if col in overrides:
            s[:] = overrides[col]
            notes.append(f"{col} overridden at {overrides[col]}")
        if s.isna().any():
            s = s.ffill()
            if s.isna().any():  # nothing to carry forward
                s = s.fillna(0.0)
                notes.append(f"{col}: no WEO projection, set to zero")
        return s

    r, g, pi, e, pb = (series(c) for c in ["effective_rate", "real_growth", "deflator_growth", "fx_depreciation", "primary_balance"])
    exp = series("expenditure") - t["interest"].reindex(years).ffill().fillna(0) if "expenditure" in t else None
    oil = series("oil_price") if "oil_price" in t else None
    # Current account is for display and BoP shocks only: no note if it's missing.
    ca = t["current_account"].reindex(years).ffill() if "current_account" in t else None

    a = Assumptions(
        initial_debt=float(t.loc[last_actual_year, "debt"]),
        effective_rate=r,
        real_growth=g,
        deflator_growth=pi,
        fx_depreciation=e,
        primary_balance=pb,
        other_flows=pd.Series(0.0, index=years),
        fx_share=fx_share,
        repricing_share=repricing_share,
        expenditure=exp,
        oil_price=oil,
        current_account=ca,
        notes=notes,
    )
    # Back out the residual year by year so the baseline matches WEO debt:
    # residual = actual change in debt - (automatic dynamics - primary balance).
    debt_prev = a.initial_debt
    for y in years:
        if y not in weo_years:
            break
        parts = automatic_dynamics(debt_prev, r[y] / 100, g[y] / 100, pi[y] / 100, e[y] / 100, fx_share / 100)
        a.other_flows[y] = float(t.loc[y, "debt"]) - debt_prev - sum(parts.values()) + pb[y]
        debt_prev = float(t.loc[y, "debt"])
    return a


# ---------------------------------------------------------------- history

def historical_drivers(table: pd.DataFrame, last_actual_year: int, start: int = 1990) -> pd.DataFrame:
    """Past values of the debt drivers (used by the historical scenario and fan chart)."""
    cols = ["effective_rate", "real_growth", "deflator_growth", "fx_depreciation", "primary_balance"]
    h = table.loc[(table.index >= start) & (table.index <= last_actual_year), cols]
    return h.dropna()


def historical_stats(table: pd.DataFrame, last_actual_year: int, window: int = 10) -> pd.DataFrame:
    """Average and standard deviation of each driver over the last `window` years."""
    h = historical_drivers(table, last_actual_year).tail(window)
    return pd.DataFrame({"mean": h.mean(), "std": h.std(ddof=1), "years": len(h)})
