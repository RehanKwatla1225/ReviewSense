from __future__ import annotations

import os
from typing import Dict, List


def _build_fallback(sentiment_counts: Dict[str, int], topic_keywords: Dict[int, List[str]], sample_reviews: List[str]) -> str:
    topics_sorted = sorted(topic_keywords.items(), key=lambda item: item[0])[:3]
    topic_lines = []
    for topic_id, words in topics_sorted:
        words_text = ', '.join(words[:6]) if words else 'general feedback'
        topic_lines.append(f'Topic {topic_id + 1}: {words_text}.')

    sample_lines = '\n'.join(f'- {s[:180]}' for s in sample_reviews[:3]) if sample_reviews else '- No sample reviews available.'

    return (
        'Executive Summary\n\n'
        f'The dataset contains {sum(sentiment_counts.values())} reviews, with '
        f'{sentiment_counts.get("Positive", 0)} positive, {sentiment_counts.get("Neutral", 0)} neutral, '
        f'and {sentiment_counts.get("Negative", 0)} negative entries.\n\n'
        'The dominant topics appear to center on the themes below.\n'
        + '\n'.join(topic_lines)
        + '\n\n'
        f'Sample reviews:\n{sample_lines}'
    )


def generate_summary(sentiment_counts: Dict[str, int], topic_keywords: Dict[int, List[str]], sample_reviews: List[str]) -> str:
    api_key = os.getenv('GROQ_API_KEY', '').strip()
    if not api_key:
        return _build_fallback(sentiment_counts, topic_keywords, sample_reviews)

    prompt = {
        'sentiment_counts': sentiment_counts,
        'topic_keywords': topic_keywords,
        'sample_reviews': sample_reviews[:20],
    }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            max_tokens=800,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a business analyst who writes concise review intelligence summaries in 3 paragraphs.',
                },
                {
                    'role': 'user',
                    'content': (
                        'Create a 3-paragraph executive summary from this review analysis data. '
                        'Mention sentiment balance, key topics, and product or service implications. '
                        f'Data: {prompt}'
                    ),
                },
            ],
        )
        text = response.choices[0].message.content.strip()
        return text or _build_fallback(sentiment_counts, topic_keywords, sample_reviews)
    except Exception:
        return _build_fallback(sentiment_counts, topic_keywords, sample_reviews)
