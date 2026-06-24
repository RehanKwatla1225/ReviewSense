from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

PREFERRED_COLUMNS = ['review', 'reviews', 'text', 'comment', 'comments', 'feedback', 'content', 'description']


def load_csv(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    return pd.read_csv(source)


def detect_review_column(df: pd.DataFrame) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in PREFERRED_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]
    for col in df.columns:
        if 'review' in str(col).lower() or 'comment' in str(col).lower() or 'feedback' in str(col).lower():
            return col
    text_cols = [c for c in df.columns if df[c].dtype == 'object']
    return text_cols[0] if text_cols else None


def clean_text(series: pd.Series) -> pd.Series:
    def _clean(value):
        if pd.isna(value):
            return ''
        text = html.unescape(str(value))
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text
    return series.apply(_clean)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)
