"""Checks for the DSA engine, shocks and fan chart (no internet needed)."""

import pandas as pd
import pytest

from dsa.engine import Assumptions, baseline_from_weo, debt_path, historical_stats
from dsa.fanchart import fan_chart
from dsa.shocks import CustomShock, custom, preset

# IMF 2021 MAC review, Table A2 "Ruritania: Baseline Scenario" (pages 52-53).
RURITANIA = pd.DataFrame(
    {
        "effective_rate": [3.9, 3.7, 4.0, 4.4, 5.2, 5.5],
        "real_growth": [0.3, 5.9, 2.1, 0.9, 1.4, 1.7],
        "deflator_growth": [-2.4, -2.3, 1.2, 1.8, 2.0, 2.0],
        "real_interest": [3.4, 3.5, 1.7, 1.6, 2.0, 2.3],
        "growth_contribution": [-0.2, -3.4, -1.3, -0.5, -0.9, -1.1],
        "automatic": [3.2, 0.1, 0.4, 1.0, 1.1, 1.2],
    },
    index=[2019, 2020, 2021, 2022, 2023, 2024],
)
RURITANIA_DEBT = {2018: 53.5, 2019: 58.7, 2020: 61.1, 2021: 62.2, 2022: 65.1, 2023: 69.3, 2024: 73.1}


@pytest.mark.parametrize("year", RURITANIA.index)
def test_matches_imf_ruritania_decomposition(year):
    """Our decomposition reproduces the IMF's published numbers (to rounding)."""
    row = RURITANIA.loc[year]
    one = pd.Series([0.0], index=[year])
    a = Assumptions(
        initial_debt=RURITANIA_DEBT[year - 1],
        effective_rate=pd.Series([row.effective_rate], index=[year]),
        real_growth=pd.Series([row.real_growth], index=[year]),
        deflator_growth=pd.Series([row.deflator_growth], index=[year]),
        fx_depreciation=one.copy(),
        primary_balance=one.copy(),
        other_flows=one.copy(),
    )
    out = debt_path(a).loc[year]
    # The IMF table shows inputs rounded to 0.1, so allow 0.15pp of rounding error.
    assert out.real_interest_rate == pytest.approx(row.real_interest, abs=0.15)
    assert out.real_gdp_growth == pytest.approx(row.growth_contribution, abs=0.15)
    assert out.automatic_debt_dynamics == pytest.approx(row.automatic, abs=0.15)


def _toy_weo(years=range(2010, 2032), last_actual=2025):
    idx = list(years)
    n = len(idx)
    t = pd.DataFrame(
        {
            "debt": [50 + 1.5 * i for i in range(n)],
            "effective_rate": [6.0] * n,
            "real_growth": [3.0 + (i % 3 - 1) for i in range(n)],
            "deflator_growth": [4.0 + (i % 4 - 1.5) for i in range(n)],
            "fx_depreciation": [5.0 * ((i % 5) - 2) for i in range(n)],
            "primary_balance": [-1.0 + 0.5 * (i % 2) for i in range(n)],
            "expenditure": [30.0] * n,
            "interest": [3.0] * n,
            "oil_price": [80.0] * n,
        },
        index=idx,
    )
    return t, last_actual


def test_baseline_reproduces_weo_debt_path():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=40)
    out = debt_path(a)
    for y in range(last + 1, 2032):
        assert out.loc[y, "debt"] == pytest.approx(t.loc[y, "debt"], abs=1e-9)
    # Beyond the WEO horizon, drivers are held flat and the residual is zero.
    assert a.other_flows[2035] == 0.0
    assert a.real_growth[2035] == a.real_growth[2031]


def test_zero_shock_changes_nothing():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=40)
    pd.testing.assert_frame_equal(debt_path(a), debt_path(custom(a, CustomShock())))


def test_shocks_push_debt_the_right_way():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=40)
    base = debt_path(a)["debt"]
    worse = {
        "growth": CustomShock(growth_pp=-3),
        "rates": CustomShock(interest_bp=300),
        "primary balance": CustomShock(primary_balance_pp=-2),
        "depreciation": CustomShock(depreciation_pct=30),
        "oil exporter, price falls": CustomShock(oil_price_pct=-30, oil_role="exporter"),
        "oil importer, price rises": CustomShock(oil_price_pct=30, oil_role="importer"),
    }
    for label, shock in worse.items():
        shocked = debt_path(custom(a, shock))["debt"]
        assert shocked.iloc[-1] > base.iloc[-1], label
    # The baseline object itself is untouched by shocks.
    pd.testing.assert_series_equal(debt_path(a)["debt"], base)


def test_depreciation_without_fx_debt_only_works_through_inflation():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=0)
    out = debt_path(custom(a, CustomShock(depreciation_pct=30, interactions=False)))
    pd.testing.assert_series_equal(out["debt"], debt_path(a)["debt"])


def test_presets_and_fan_chart_run():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=40)
    stats = historical_stats(t, last)
    base_end = debt_path(a)["debt"].iloc[-1]
    for name in ["growth", "primary_balance", "depreciation", "interest", "contingent", "natural_disaster", "combined"]:
        assert debt_path(preset(name, a, stats))["debt"].iloc[-1] > base_end, name
    fan = fan_chart(a, t.loc[:last], n=2000)
    assert list(fan.standard.columns)[0] == "p5"
    assert (fan.standard["p95"] >= fan.standard["p5"]).all()
    assert fan.width > 0
    assert 0 <= fan.prob_not_stabilising <= 100


def test_depreciation_raises_foreign_currency_debt_service():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=60)
    base = debt_path(a)
    shocked = debt_path(custom(a, CustomShock(depreciation_pct=30, interactions=False)))
    y = a.years[1]
    # Interest and repayments on foreign-currency debt cost 30% more in local currency.
    assert shocked.loc[y, "gross_financing_need"] > base.loc[y, "gross_financing_need"] + 1


def test_current_account_shock_depreciates_and_raises_gfn():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=50)
    s = CustomShock(current_account_pp=-2, depreciation_per_ca_pp=5, interactions=False)
    shocked = custom(a, s)
    y = a.years[1]
    assert shocked.fx_depreciation[y] == pytest.approx(a.fx_depreciation[y] + 10)
    assert debt_path(shocked).loc[y, "gross_financing_need"] > debt_path(a).loc[y, "gross_financing_need"]


def test_oil_moves_current_account_by_net_oil_exports():
    from dsa.shocks import current_account_change

    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=30)
    importer = CustomShock(oil_price_pct=50, oil_role="importer", net_oil_exports=-4)
    change = current_account_change(a, importer)
    assert change[a.years[1]] == pytest.approx(-2.0)  # -4% of GDP x 50%
    assert change[a.years[0]] == 0.0


def test_external_financing_stop_shortens_maturity():
    t, last = _toy_weo()
    a = baseline_from_weo(t, last, fx_share=30, repricing_share=20)
    s = CustomShock(external_stop=True, foreign_private_share=30, foreign_rollover_pct=67, stop_years=(2, 3))
    shocked = custom(a, s)
    y2, y3, y4, y5 = a.years[1:5]
    new = (1 - 0.67) * 0.30 * 0.20 * 100  # % of debt replaced each stress year
    assert shocked.extra_maturing[y3] == pytest.approx(new)
    assert shocked.extra_maturing[y4] == pytest.approx(2 * new)
    assert shocked.extra_maturing[y5] == 0.0
    out, base = debt_path(shocked), debt_path(a)
    assert out.loc[y4, "gross_financing_need"] > base.loc[y4, "gross_financing_need"]
