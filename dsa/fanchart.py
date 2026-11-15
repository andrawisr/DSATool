"""Debt fan chart, following Annex VI of the IMF's 2021 MAC SRDSF review.

In plain English: we replay the country's own history of shocks. We pick a
random pair of consecutive past years and apply that pair's growth, interest,
inflation, exchange rate and primary balance surprises to the next two
projection years, then pick another pair, and so on out to year 5. Repeating
this 10,000 times gives a cloud of possible debt paths. The percentiles of
that cloud are the fan chart.

  * Historical fan: uses the past values themselves, so it ignores the
    baseline forecast. If the baseline sits below its 20th percentile, the
    IMF treats the baseline as possibly too optimistic.
  * Standard fan: uses past *surprises* (past values minus their average),
    added to the baseline, so it is centred on the forecast.

As in the IMF method, the foreign-currency share is held fixed and there is
no feedback from debt to interest rates, so the fan understates the most
explosive paths. Not replicated: the uncertainty about starting debt (from
WEO revisions), and the "realism-adjusted" skewed fan, which needs the IMF's
cross-country database.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import Assumptions

COLS = ["effective_rate", "real_growth", "deflator_growth", "fx_depreciation", "primary_balance"]
PERCENTILES = [5, 10, 20, 25, 50, 75, 80, 90, 95]


@dataclass
class FanChart:
    years: list[int]
    baseline: pd.Series  # baseline debt path
    standard: pd.DataFrame  # percentiles (columns) by year (rows), centred on the baseline
    historical: pd.DataFrame  # percentiles of the history-only fan
    width: float  # 95th minus 5th percentile of debt in the final year (pp of GDP)
    prob_not_stabilising: float  # % of draws where the baseline primary balance in the final year is below the debt-stabilising one
    baseline_below_hist_p20: bool  # IMF realism flag
    draws: int
    history_years: tuple[int, int]


def _block_draws(n_hist: int, horizon: int, n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices into the history, drawn in consecutive blocks (shape n x horizon)."""
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, n_hist - block + 1, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)
    return idx[:, :horizon]


def _simulate(debt0: float, drivers: np.ndarray, sfa: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised debt dynamics. drivers: (n, T, 5) in percent. Returns debt (n, T) and stabilising PB (n, T)."""
    r, g, pi, e, pb = (drivers[:, :, i] / 100 for i in range(5))
    pb = pb * 100
    n, T = r.shape
    debt = np.empty((n, T))
    stab = np.empty((n, T))
    prev = np.full(n, debt0)
    for t in range(T):
        denom = (1 + g[:, t]) * (1 + pi[:, t])
        auto = (r[:, t] - pi[:, t] * (1 + g[:, t]) - g[:, t] + alpha * e[:, t] * (1 + r[:, t])) / denom * prev
        stab[:, t] = auto + sfa[t]
        prev = prev + auto - pb[:, t] + sfa[t]
        debt[:, t] = prev
    return debt, stab


def fan_chart(
    baseline: Assumptions,
    history: pd.DataFrame,
    horizon: int = 5,
    n: int = 10_000,
    block: int = 2,
    seed: int = 1,
) -> FanChart:
    hist = history[COLS].dropna().to_numpy(dtype=float)
    if len(hist) < 2 * block + 2:
        raise ValueError("Not enough history for a fan chart (need at least 6 years of complete data).")
    years = baseline.years[:horizon]
    base = np.column_stack([getattr(baseline, c).reindex(years).to_numpy(dtype=float) for c in COLS])
    sfa = baseline.other_flows.reindex(years).to_numpy(dtype=float)
    alpha = baseline.fx_share / 100

    rng = np.random.default_rng(seed)
    idx = _block_draws(len(hist), horizon, n, block, rng)
    draws_hist = hist[idx]  # (n, T, 5)
    surprises = draws_hist - hist.mean(axis=0)

    std_debt, std_stab = _simulate(baseline.initial_debt, base[None] + surprises, sfa, alpha)
    hist_debt, _ = _simulate(baseline.initial_debt, draws_hist, np.zeros(horizon), alpha)
    base_debt, _ = _simulate(baseline.initial_debt, base[None], sfa, alpha)
    base_path = pd.Series(base_debt[0], index=years)

    def pct(x):
        return pd.DataFrame(np.percentile(x, PERCENTILES, axis=0).T, index=years, columns=[f"p{p}" for p in PERCENTILES])

    std_p, hist_p = pct(std_debt), pct(hist_debt)
    pb_final = base[-1, COLS.index("primary_balance")]
    return FanChart(
        years=years,
        baseline=base_path,
        standard=std_p,
        historical=hist_p,
        width=float(std_p["p95"].iloc[-1] - std_p["p5"].iloc[-1]),
        prob_not_stabilising=float((pb_final < std_stab[:, -1]).mean() * 100),
        baseline_below_hist_p20=bool((base_path < hist_p["p20"]).any()),
        draws=n,
        history_years=(int(history.index.min()), int(history.index.max())),
    )
