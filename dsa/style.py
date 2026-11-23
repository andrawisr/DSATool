"""Look and feel: a research-report style (navy and teal, serif headings).

In plain English: this file only changes how the app looks, not any numbers.
To go back to Streamlit's plain look, delete the two lines in app.py that call
`apply()` and `hero()`, or undo the commit that added this file.

Fonts: the headings use a serif already on your Mac (Iowan Old Style, then
Palatino or Georgia), and the body uses Streamlit's built-in Source Sans, so the
app still makes no requests to font servers.
"""

from __future__ import annotations

import html

import plotly.io as pio
import streamlit as st

NAVY = "#102547"
SLATE = "#2A3544"
TEAL = "#36738A"
GREY = "#656C77"
PANEL = "#F5F7FB"
RULE = "#DDE2EA"
SERIF = '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif'

# Line colours for the charts: navy and teal first, then muted colours that
# still tell the stress tests apart.
COLOURWAY = [NAVY, TEAL, "#B5533C", "#B08D57", "#5E8C6A", "#6B5B95", "#7FB2C4", GREY, "#C9A227", "#8C4A6B"]
CHART_COLOURS = {"baseline": NAVY, "custom": "#B5533C", "historical": GREY}
FAN_RGB = "16,37,71"  # navy, for the fan chart bands

CSS = f"""
<style>
h1, h2, h3, h4, [data-testid="stHeading"] * {{
    font-family: {SERIF} !important;
    font-weight: 400 !important;
    color: {SLATE};
    letter-spacing: 0;
}}
p, li, label, [data-testid="stCaptionContainer"] {{ color: {GREY}; }}
[data-testid="stMarkdownContainer"] p {{ line-height: 1.5; }}

/* Header band */
.ls-hero {{
    background: linear-gradient(120deg, #0B1E2E 0%, {NAVY} 55%, #1F4F5C 100%);
    color: #fff;
    padding: 2.2rem 2.4rem 2rem;
    margin: 0 0 1.4rem;
}}
.ls-hero .eyebrow {{
    font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: rgba(255,255,255,0.75); margin-bottom: 0.4rem;
}}
.ls-hero .title {{
    font-family: {SERIF}; font-weight: 400; font-size: 2.9rem; line-height: 1.1; color: #fff;
}}
.ls-hero .sub {{ font-size: 0.95rem; color: rgba(255,255,255,0.8); margin-top: 0.7rem; max-width: 52rem; }}

/* Headline numbers: large serif figures over small labels */
[data-testid="stMetric"] {{ border-top: 1px solid {RULE}; padding-top: 0.6rem; }}
[data-testid="stMetricValue"] {{ font-family: {SERIF}; font-weight: 400; color: {SLATE}; font-size: 2.3rem; }}
[data-testid="stMetricLabel"] p {{ color: {GREY}; font-size: 0.85rem; }}

/* Tabs: teal underline on the selected tab */
button[data-baseweb="tab"] p {{ font-size: 0.95rem; color: {GREY}; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color: {NAVY}; }}
[data-baseweb="tab-highlight"] {{ background-color: {TEAL} !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] h1 {{ color: {NAVY}; font-size: 1.9rem; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {SLATE}; font-size: 1.25rem; }}
[data-testid="stExpander"] details {{ border-color: {RULE}; background: #fff; }}

a {{ color: {TEAL}; }}
</style>
"""


def apply() -> None:
    """Add the styles to the page and make them the default for charts."""
    st.markdown(CSS, unsafe_allow_html=True)
    template = pio.templates["plotly_white"]
    template.layout.colorway = COLOURWAY
    template.layout.font = dict(family='"Source Sans Pro", "Source Sans 3", sans-serif', color=SLATE, size=13)
    template.layout.xaxis.gridcolor = RULE
    template.layout.yaxis.gridcolor = RULE
    pio.templates.default = "plotly_white"


def hero(title: str, subtitle: str = "", eyebrow: str = "Debt sustainability analysis") -> None:
    """The dark header band at the top of the page."""
    st.markdown(
        f'<div class="ls-hero"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="title">{html.escape(title)}</div>'
        + (f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )
