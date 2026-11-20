"""Build the Article IV download checklist (article_iv/Article IV checklist.xlsx).

In plain English: for each country in the app's dropdown, this shows how good
the automatic figures are, where to find the IMF's debt analysis, and whether
you've already added Article IV figures. Countries whose automatic figures are
weakest come first.

Run it again whenever you add Article IV figures or save new PDFs:

    .venv/bin/python -m dsa.data.checklist

Anything you type in the "Your notes" column is kept when it's rebuilt.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import fx_share, maturity, overrides

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "article_iv"
CHECKLIST_PATH = FOLDER / "Article IV checklist.xlsx"
COVERED_PATH = Path(__file__).with_name("countries_covered.csv")
# Links to each country's latest Article IV, read from the IMF country pages
# (imf.org/en/countries/<code>). Add or fix a row by hand when a new report is out.
LINKS_PATH = Path(__file__).with_name("article_iv_links.csv")
IMF_PAGE_CODE = {"KOS": "UVK"}  # the IMF's own code where it differs

# Countries eligible for the IMF's concessional lending (PRGT) whose debt
# analysis uses the joint IMF-World Bank low-income framework (LIC-DSF 2018).
# Their DSA is a separate document, not an annex. Based on recent IMF reports;
# the report itself says which framework it uses.
LIC_DSF = {
    "BEN", "BFA", "BGD", "CIV", "CMR", "COD", "COG", "CPV", "ETH", "GHA", "HND", "KEN",
    "KGZ", "KHM", "LAO", "LSO", "MDA", "MDG", "MDV", "MOZ", "NER", "NIC", "PNG", "RWA",
    "SEN", "TGO", "TJK", "TZA", "UGA", "UZB", "ZMB",
}

NO_RECENT_ARTICLE_IV = {
    "TWN": "Not an IMF member, so there is no Article IV. Use the authorities' own debt data.",
    "VEN": "No Article IV since 2004. Use the authorities' or the market's figures.",
    "RUS": "No Article IV since 2021, so the latest figures are old.",
}

# Countries you've marked as having no DSA to use. They keep the automatic estimates.
NO_DSA = {
    "SEN": "No DSA available. The app uses the automatic estimates.",
    "BHR": "No DSA available. The app uses the automatic estimates.",
}

FX_LABEL = {
    "override": "Article IV",
    "qpsd_currency": "World Bank currency split (good)",
    "rule_advanced": "Rule: own-currency borrower",
    "rule_dollarised": "Rule: dollarised",
    "rule_euro": "Rule: euro area",
    "ids_external": "External debt stand-in (rough)",
    "missing": "None",
}
MAT_LABEL = {
    "override": "Article IV",
    "qpsd_due_12m": "World Bank debt due in 12 months (good)",
    "external_service_minus_interest": "External debt service + peer domestic rate (rough)",
    "default": "None: 25% default",
}

PRIORITY_TEXT = {
    1: "1 - No automatic figure",
    2: "2 - Rough foreign-currency share",
    3: "3 - Rough maturity",
    4: "4 - Already good (optional)",
    5: "Done",
}
PRIORITY_FILL = {1: "F4CCCC", 2: "FCE5CD", 3: "FFF2CC", 4: "D9EAD3", 5: "CFE2F3"}

HEADERS = [
    "Priority", "Country", "Code", "Latest report (link)", "Published", "IMF country page (link)",
    "Where to find the DSA", "Status",
    "Foreign-currency share now (%)", "Foreign-currency share source",
    "Maturing share now (%)", "Maturing share source",
    "What to take from the report", "Your notes",
]


def _pdf_for(iso3: str, name: str, pdfs: list[Path]) -> Path | None:
    first_word = name.split(",")[0].split()[0].lower()
    if first_word in ("congo", "korea"):  # ambiguous first words: need the full name or code
        first_word = name.lower()
    for p in pdfs:
        stem = p.stem.lower()
        if stem.startswith(iso3.lower()) or name.lower() in stem or stem.startswith(first_word):
            return p
    return None


def build_rows() -> pd.DataFrame:
    covered = pd.read_csv(COVERED_PATH)
    # Reports saved in article_iv/ or "Article IVs/" (with or without a .pdf ending).
    pdfs = sorted(p for d in (FOLDER, ROOT / "Article IVs") if d.exists() for p in d.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in ("", ".pdf"))
    links = pd.read_csv(LINKS_PATH, dtype=str).set_index("iso3") if LINKS_PATH.exists() else pd.DataFrame()
    rows = []
    for iso3, name in zip(covered["iso3"], covered["name"]):
        name = name.strip()
        fx = fx_share.lookup(iso3)
        mat = maturity.lookup(iso3)
        has = {f: overrides.get(iso3, f) is not None for f in ("fx_share", "maturing_share", "foreign_private_share")}

        if has["fx_share"] and has["maturing_share"]:
            prio = 5
        elif fx["method"] == "missing" or mat["method"] == "default":
            prio = 1
        elif fx["method"] == "ids_external" and not has["fx_share"]:
            prio = 2
        elif mat["method"] == "external_service_minus_interest" and not has["maturing_share"]:
            prio = 3
        else:
            prio = 4

        want = []
        if not has["fx_share"]:
            want.append("foreign-currency share of debt")
        if not has["maturing_share"]:
            want.append("amortisation / share of debt maturing each year")
        if not has["foreign_private_share"]:
            want.append("share held by foreign private creditors")

        if iso3 in NO_RECENT_ARTICLE_IV:
            where = NO_RECENT_ARTICLE_IV[iso3]
        elif iso3 in NO_DSA:
            where = NO_DSA[iso3]
        elif iso3 in LIC_DSF:
            where = "Separate joint IMF-World Bank 'Debt Sustainability Analysis' (low-income framework)"
        else:
            where = "Article IV staff report, annex 'Sovereign Risk and Debt Sustainability Analysis'"

        pdf = _pdf_for(iso3, name, pdfs)
        if prio == 5:
            status = "Article IV figures added" + ("" if has["foreign_private_share"] else " (foreign private share still missing)")
        elif any(has.values()):
            status = "Partly added: " + ", ".join(f for f, v in has.items() if v)
        elif iso3 in NO_RECENT_ARTICLE_IV:
            status = "No recent Article IV (see 'Where to find the DSA')"
        elif iso3 in NO_DSA:
            status = "No DSA"
        elif pdf is not None:
            status = f"PDF saved ({pdf.name}): send it to Claude"
        else:
            status = "To download"

        rows.append({
            "prio": prio,
            "Priority": PRIORITY_TEXT[prio],
            "Country": name,
            "Code": iso3,
            "Latest report (link)": links.loc[iso3, "report"] if iso3 in links.index else "",
            "_report_url": links.loc[iso3, "url"] if iso3 in links.index else "",
            "Published": links.loc[iso3, "published"] if iso3 in links.index else "",
            "IMF country page (link)": "" if iso3 == "TWN" else "Open country page",
            "_page_url": "" if iso3 == "TWN" else f"https://www.imf.org/en/countries/{IMF_PAGE_CODE.get(iso3, iso3).lower()}",
            "Where to find the DSA": where,
            "Status": status,
            "Foreign-currency share now (%)": fx["fx_share"],
            "Foreign-currency share source": FX_LABEL.get(fx["method"], fx["method"]),
            "Maturing share now (%)": mat["maturing_share"],
            "Maturing share source": MAT_LABEL.get(mat["method"], mat["method"]),
            "What to take from the report": "; ".join(want) if want else "Nothing: refresh when a new Article IV comes out",
        })
    return pd.DataFrame(rows).sort_values(["prio", "Country"]).reset_index(drop=True)


def _old_notes() -> dict[str, str]:
    if not CHECKLIST_PATH.exists():
        return {}
    try:
        old = pd.read_excel(CHECKLIST_PATH, sheet_name="Checklist", header=0)
        return {r["Code"]: r["Your notes"] for _, r in old.iterrows() if isinstance(r.get("Your notes"), str)}
    except Exception:
        return {}


def write_checklist() -> Path:
    FOLDER.mkdir(exist_ok=True)
    rows = build_rows()
    notes = _old_notes()
    base = Font(name="Arial", size=10)
    bold = Font(name="Arial", size=10, bold=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Checklist"
    ws.append(HEADERS)
    for c in ws[1]:
        c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="434343")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for _, r in rows.iterrows():
        ws.append([r[h] if h != "Your notes" else notes.get(r["Code"], "") for h in HEADERS])
        i = ws.max_row
        for c in ws[i]:
            c.font = base
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(i, 1).fill = PatternFill("solid", fgColor=PRIORITY_FILL[r["prio"]])
        for col, key in ((4, "_report_url"), (6, "_page_url")):
            if r[key]:
                ws.cell(i, col).hyperlink = r[key]
                ws.cell(i, col).font = Font(name="Arial", size=10, color="0563C1", underline="single")
        if not r["_report_url"] and r["_page_url"]:
            ws.cell(i, 4).value = "Not collected: use the country page"
        for col in (9, 11):
            ws.cell(i, col).number_format = "0.0"
        notes_cell = ws.cell(i, len(HEADERS))
        notes_cell.fill = PatternFill("solid", fgColor="FFFF00")
    widths = [26, 24, 7, 30, 11, 18, 44, 34, 14, 28, 12, 34, 44, 30]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{ws.max_row}"

    # How-to sheet with live counts.
    hs = wb.create_sheet("How to use", 0)
    last = ws.max_row
    lines = [
        ("Article IV download checklist", bold),
        ("", base),
        ("1. Work down the Checklist tab from the top: priority 1 countries have no automatic figure at all.", base),
        ("2. Click 'Latest report (link)', or 'IMF country page' where no direct link was collected, and use 'Download PDF'.", base),
        ("3. Save it in this folder (dsa-tool/article_iv), named with the country code first, e.g. 'SAU 2025.pdf'.", base),
        ("4. Send Claude a batch of PDFs. Claude adds the figures to dsa/data/country_overrides.csv and shows old vs new.", base),
        ("5. Rebuild this file to update the Status column: .venv/bin/python -m dsa.data.checklist", base),
        ("", base),
        ("The only column to edit is 'Your notes' (yellow). It is kept when the file is rebuilt; everything else is overwritten.", base),
        ("Example note: 'Downloaded 2025 Article IV, 12 Mar; DSA annex on pp. 58-70'.", base),
        ("", base),
        ("Where to find the DSA: emerging and advanced economies have an annex in the Article IV staff report.", base),
        ("Low-income countries have a separate joint IMF-World Bank DSA, often published with a programme review.", base),
        ("The low-income list is based on recent IMF reports; the report itself says which framework it uses.", base),
        ("Direct links were collected in November 2026 for 48 countries before imf.org stopped answering automated requests.", base),
        ("The country page always lists the newest Article IV at the top, so it stays current when new reports come out.", base),
        ("", base),
        ("Countries by priority", bold),
    ]
    for text, font in lines:
        hs.append([text])
        hs.cell(hs.max_row, 1).font = font
    for p in range(1, 6):
        hs.append([PRIORITY_TEXT[p], f'=COUNTIF(Checklist!$A$2:$A${last},"{PRIORITY_TEXT[p]}")'])
        hs.cell(hs.max_row, 1).font = base
        hs.cell(hs.max_row, 1).fill = PatternFill("solid", fgColor=PRIORITY_FILL[p])
        hs.cell(hs.max_row, 2).font = base
    hs.append(["Total", f"=SUM(B{hs.max_row - 4}:B{hs.max_row})"])
    hs.cell(hs.max_row, 1).font = bold
    hs.cell(hs.max_row, 2).font = bold
    hs.column_dimensions["A"].width = 40
    hs.column_dimensions["B"].width = 10

    wb.calculation.fullCalcOnLoad = True  # Excel/Numbers fill in the counts on opening
    wb.save(CHECKLIST_PATH)
    return CHECKLIST_PATH


if __name__ == "__main__":
    path = write_checklist()
    print(f"Saved {path}")
