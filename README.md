# dsa-tool

A private, IMF-style debt sustainability analysis (DSA) for any country. It runs only on your own computer.

## Start the app

In Terminal:

```bash
cd ~/Documents/dsa-tool
.venv/bin/streamlit run app.py
```

Your browser opens at http://localhost:8501. To stop the app, press Ctrl+C in Terminal.

## What's inside

| Piece | File | What it does |
|---|---|---|
| Data | `dsa/data/weo.py` | Downloads IMF World Economic Outlook history and projections for a country, keeps a local copy for a week, and works out the effective interest rate (interest ÷ last year's debt) and the exchange rate (GDP in local currency ÷ GDP in US$). |
| Foreign-currency share | `dsa/data/fx_share.py` → `fx_share_estimates.csv` | An estimate of the share of each country's government debt in foreign currency. Sources, best first: World Bank Quarterly Public Sector Debt; external public debt from World Bank IDS (via World Development Indicators); rules for the euro area, advanced economies, and countries that use the US dollar or euro. You can always override it in the app. Rebuild with `.venv/bin/python -m dsa.data.fx_share`. |
| Engine | `dsa/engine.py` | The IMF debt dynamics equation and decomposition (MAC SRDSF 2021, Table A2 note 5). The baseline reproduces the WEO debt path, with the gap carried as a "residual". |
| Shocks | `dsa/shocks.py` | Your custom shocks plus IMF standard stress tests (LIC-DSF Guidance Note 2018, Table 8; MAC SRDSF 2021, Table 5), with the IMF's knock-on effects. |
| Fan chart | `dsa/fanchart.py` | 10,000 simulated debt paths built from two-year blocks of the country's own historical shocks (MAC SRDSF 2021, Annex VI), with the 20th-percentile realism check. |
| Screen | `app.py` | The Streamlit app. |

## Tests

```bash
.venv/bin/python -m pytest
```

Tests include checking the engine against the IMF's worked example ("Ruritania", 2021 MAC review). Add `-m "not network"` to skip the tests that download live data.

## Known limitations

- WEO reports *net* interest, which understates the effective rate for countries with large financial assets. Use the override in the sidebar.
- Gross financing needs approximate amortisation as a fixed share of debt (the "repricing" slider), because WEO has no maturity data.
- Not replicated: the IMF's mechanical risk signals (they need coefficients from the 2022 Staff Guidance Note), the GFN financeability module (needs creditor holdings data), the skewed "realism-adjusted" fan chart, and the LIC-DSF external DSA (needs present values of external debt and export data).
