"""Country figures taken from IMF Article IV reports (or your own research).

In plain English: the automatic estimates (foreign-currency share, share of
debt maturing each year) come from World Bank data and can be off for a given
country. When you have a better number, for example from the DSA annex of an
Article IV, add a row to `country_overrides.csv`:

    iso3, field, value, source, as_of

`field` is `fx_share`, `maturing_share`, `foreign_private_share` or `net_oil_exports`. Overrides always win over the
automatic estimates, and the app shows their source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OVERRIDES_PATH = Path(__file__).with_name("country_overrides.csv")


def get(iso3: str, field: str) -> dict | None:
    """The override for one country and field, or None."""
    if not OVERRIDES_PATH.exists():
        return None
    t = pd.read_csv(OVERRIDES_PATH, dtype={"as_of": str}).fillna({"as_of": ""})
    row = t[(t["iso3"] == iso3.upper()) & (t["field"] == field)]
    if row.empty:
        return None
    r = row.iloc[-1]
    return {"value": float(r["value"]), "source": r["source"], "as_of": r["as_of"], "quality": "Article IV"}
