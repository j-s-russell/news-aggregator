from html import escape
from pathlib import Path

import pandas as pd

import streamlit as st

CSS_PATH = Path(__file__).parent / "styles.css"


def inject_css():
    st.html(f"<style>{CSS_PATH.read_text()}</style>")


def get_bias_color(val):
    if val <= -2:
        return "#2563EB"  # left - blue
    elif val == -1:
        return "#60A5FA"  # center-left - light blue
    elif val == 0:
        return "#9CA3AF"  # center - gray
    elif val == 1:
        return "#FCA5A5"  # center-right - light red
    elif val >= 2:
        return "#DC2626"  # right - red


def bias_bar_html(val, show_scale=False):
    labels = (
        "<span>Left</span><span>Source bias</span><span>Right</span>"
        if show_scale else "<span>Source bias</span>"
    )
    return (
        f'<div class="bias-bar-wrap{" wide" if show_scale else ""}">'
        f'<div class="bias-gradient">'
        f'<div class="bias-marker" style="left: {50 + val * 25}%; background: {get_bias_color(val)};"></div>'
        f'</div>'
        f'<div class="bias-label">{labels}</div>'
        f'</div>'
    )


def has_value(val):
    return val is not None and not (isinstance(val, float) and pd.isna(val)) and str(val).strip().lower() not in ("", "nan", "none")


def author_html(row):
    return f'<span class="card-author">By {escape(str(row["author"]))}</span>' if has_value(row.get("author")) else ""


def meta_html(row, show_topic=True):
    parts = [
        f'<span class="card-source">{escape(str(row["source"]))}</span>',
        '<span class="dot">·</span>',
        f'<span>{row["publish_date"].strftime("%b %d, %Y · %H:%M")}</span>',
    ]
    if show_topic and has_value(row.get("cluster_label")):
        parts.append(f'<span class="chip accent">{escape(str(row["cluster_label"]))}</span>')
    return f'<div class="card-meta">{"".join(parts)}</div>'


def article_card_html(row, show_topic=True):
    return (
        f'<div class="card-title">{escape(str(row["title"]))}</div>'
        f'{meta_html(row, show_topic)}'
        f'{author_html(row)}'
        f'{bias_bar_html(row["source_bias"])}'
    )


def similar_row_html(row):
    return (
        f'<div class="sim-title">{escape(str(row["title"]))}</div>'
        f'<div class="sim-source">{escape(str(row["source"]))}</div>'
    )


def page_header_html(title, subtitle="", eyebrow=None):
    eyebrow_html = f'<div class="eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    return f'<div class="page-header">{eyebrow_html}<h1>{escape(str(title))}</h1>{subtitle_html}</div>'


def section_label(text):
    st.html(f'<div class="section-label">{escape(text)}</div>')
