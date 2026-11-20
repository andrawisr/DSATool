"""Checks for the WEO data layer.

The offline tests check our own arithmetic on made-up numbers. The `network`
tests download real data from the IMF; skip them with `pytest -m "not network"`.
"""

import math

import pandas as pd
import pytest

from dsa.data.weo import add_derived_series, get_country


def test_derived_series_arithmetic():
    t = pd.DataFrame(
        {
            "debt": [50.0, 55.0],
            "primary_balance": [-1.0, -2.0],
            "overall_balance": [-3.0, -4.5],
            "ngdp_lcu": [1000.0, 1100.0],
            "ngdp_usd": [200.0, 200.0],
            "deflator_index": [100.0, 104.0],
        },
        index=[2024, 2025],
    )
    out = add_derived_series(t)
    # Interest = PB - OB = -2 - (-4.5) = 2.5% of GDP.
    assert out.loc[2025, "interest"] == pytest.approx(2.5)
    # Effective rate = 2.5% * 1100 / (50% * 1000) = 27.5 / 500 = 5.5%.
    assert out.loc[2025, "effective_rate"] == pytest.approx(5.5)
    assert out.loc[2025, "nominal_growth"] == pytest.approx(10.0)
    assert out.loc[2025, "deflator_growth"] == pytest.approx(4.0)
    # FX went from 5.0 to 5.5 LCU per US$: a 10% depreciation.
    assert out.loc[2025, "fx_lcu_per_usd"] == pytest.approx(5.5)
    assert out.loc[2025, "fx_depreciation"] == pytest.approx(10.0)


@pytest.mark.network
@pytest.mark.parametrize("iso3", ["BRA", "TUR", "NGA", "SAU"])
def test_real_country_download(iso3, tmp_path):
    data = get_country(iso3, start=2010, cache_dir=tmp_path)
    t = data.table
    assert data.last_actual_year is not None
    assert len(data.projection_years) >= 5, "WEO should give at least 5 projection years"
    for col in ["debt", "primary_balance", "real_growth", "effective_rate", "fx_lcu_per_usd", "oil_price"]:
        recent = t.loc[data.last_actual_year, col]
        assert not math.isnan(recent), f"{iso3}: {col} missing in {data.last_actual_year}"
    # Sanity ranges for the effective interest rate.
    assert -5 < t.loc[data.last_actual_year, "effective_rate"] < 60

    # Second call is served from the cache with identical numbers.
    again = get_country(iso3, start=2010, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(t, again.table, check_dtype=False, check_names=False)


def test_fx_share_table_lookup():
    from dsa.data.fx_share import lookup

    assert lookup("BRA")["fx_share"] < 15  # Brazil borrows mostly in reais
    assert lookup("DEU")["fx_share"] == 0.0  # euro is Germany's own currency
    assert lookup("XXX")["fx_share"] is None  # unknown country -> no estimate


def test_fiscal_year_markers():
    from dsa.data.weo import parse_year

    assert parse_year("2025") == 2025
    assert parse_year(2025.0) == 2025
    assert parse_year("FY2024/25", "July/June") == 2025  # Egypt: filed under the year it ends
    assert parse_year("FY2024/25", "October/September") == 2025  # Thailand
    assert parse_year("FY2024/25", "April/March") == 2024  # India: filed under the year it starts
    assert parse_year("FY2024/25") == 2024  # months unknown: start year
    assert parse_year("FY2025/2026", "April/March") == 2025
    assert parse_year("n/a") is None


def test_maturity_table_lookup():
    from dsa.data.maturity import DEFAULT_SHARE, lookup

    deu = lookup("DEU")  # a country with no Article IV figures
    assert 10 < deu["maturing_share"] < 50 and deu["quality"] == "high"
    unknown = lookup("XXX")
    assert unknown["maturing_share"] == DEFAULT_SHARE and unknown["quality"] == "none"


def test_article_iv_overrides_win():
    from dsa.data import fx_share, maturity

    egy_fx, egy_mat = fx_share.lookup("EGY"), maturity.lookup("EGY")
    assert egy_fx["method"] == "override" and egy_fx["fx_share"] == 40.0
    assert egy_mat["method"] == "override" and egy_mat["maturing_share"] == 40.0
    assert egy_mat["automatic_estimate"] < 30  # the World Bank-based estimate it replaced
    assert fx_share.lookup("DEU")["method"] != "override"
