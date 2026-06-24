from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def cluster_topics(texts: List[str], n_topics: int = 5, top_words: int = 8):
    if not texts:
        return pd.DataFrame({'topic_id': []}), {'topic_keywords': {}}

    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    matrix = vectorizer.fit_transform(texts)

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        topic_ids = [0] * len(texts)
        return pd.DataFrame({'topic_id': topic_ids, 'topic_keywords': [''] * len(texts)}), {'topic_keywords': {0: []}}

    topics = max(1, min(n_topics, matrix.shape[0]))
    lda = LatentDirichletAllocation(n_components=topics, random_state=42)
    doc_topics = lda.fit_transform(matrix)
    topic_ids = doc_topics.argmax(axis=1)

    vocab = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-top_words:][::-1]
        topic_keywords[idx] = [vocab[i] for i in top_idx]

    topic_label_strings = [', '.join(topic_keywords.get(tid, [])) for tid in topic_ids]
    df = pd.DataFrame({'topic_id': topic_ids, 'topic_keywords': topic_label_strings})
    return df, {'topic_keywords': topic_keywords}
