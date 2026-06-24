from __future__ import annotations

from functools import lru_cache
from typing import List

import pandas as pd

LABEL_MAP = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}


@lru_cache(maxsize=1)
def _get_pipeline():
    try:
        from transformers import pipeline
        return pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
    except Exception:
        return None


def _fallback_label(text: str) -> tuple[str, float]:
    text = (text or '').lower()
    positive = ['good', 'great', 'excellent', 'love', 'amazing', 'happy', 'awesome', 'fast']
    negative = ['bad', 'terrible', 'awful', 'hate', 'slow', 'broken', 'poor', 'worst']
    pos = sum(word in text for word in positive)
    neg = sum(word in text for word in negative)
    if pos > neg:
        return 'Positive', min(0.99, 0.60 + 0.10 * pos)
    if neg > pos:
        return 'Negative', min(0.99, 0.60 + 0.10 * neg)
    return 'Neutral', 0.55


def analyze_sentiments(texts: List[str], batch_size: int = 32) -> pd.DataFrame:
    pipe = _get_pipeline()
    labels = []
    scores = []

    if pipe is None:
        for text in texts:
            label, score = _fallback_label(text)
            labels.append(label)
            scores.append(score)
        return pd.DataFrame({'sentiment_label': labels, 'sentiment_score': scores})

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = pipe(batch, truncation=True)
        for result in results:
            label = LABEL_MAP.get(result['label'], result['label'])
            labels.append(label)
            scores.append(round(float(result['score']), 4))

    return pd.DataFrame({'sentiment_label': labels, 'sentiment_score': scores})
