from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from modules.ai_summary import generate_summary
from modules.preprocessor import clean_text, deduplicate, detect_review_column, load_csv
from modules.sentiment import analyze_sentiments
from modules.topics import cluster_topics

load_dotenv()

ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / os.getenv('UPLOAD_FOLDER', 'uploads')
UPLOAD_DIR.mkdir(exist_ok=True)
STATE_FILE = UPLOAD_DIR / 'reviewsense_state.json'

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH_MB', '16')) * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _empty_state() -> dict[str, Any]:
    return {
        'source_file': None,
        'processed_file': None,
        'review_column': None,
        'rows': [],
        'summary': '',
        'stats': {},
        'charts': {},
        'wordcloud': None,
        'topics': {},
    }


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.replace({pd.NA: None}).to_dict(orient='records')


def _build_wordcloud_b64(text_series: pd.Series) -> str | None:
    text = ' '.join(text_series.dropna().astype(str).tolist()).strip()
    if not text:
        return None
    wc = WordCloud(width=1200, height=600, background_color='white', colormap='Blues').generate(text)
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    buffer = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def _save_state(state: dict[str, Any]) -> None:
    serializable = dict(state)
    serializable['topics'] = {str(k): v for k, v in state.get('topics', {}).items()}
    STATE_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding='utf-8')


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return _empty_state()
    try:
        payload = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        payload['topics'] = {int(k): v for k, v in payload.get('topics', {}).items()}
        return payload
    except Exception:
        return _empty_state()


def process_reviews(file_path: Path) -> dict[str, Any]:
    df = load_csv(file_path)
    review_col = detect_review_column(df)
    if review_col is None:
        raise ValueError(
            'Could not detect a review text column. Please rename it to review, reviews, text, comment, or feedback.'
        )

    df = deduplicate(df)
    df = df.copy()
    df[review_col] = clean_text(df[review_col])
    df = df[df[review_col].astype(str).str.len() > 0].reset_index(drop=True)

    sentiments = analyze_sentiments(df[review_col].tolist())
    df = pd.concat([df, sentiments], axis=1)

    topics_df, topic_model_info = cluster_topics(
        df[review_col].tolist(),
        n_topics=int(os.getenv('DEFAULT_TOPICS', '5')),
    )
    df = pd.concat([df, topics_df], axis=1)

    sentiment_counts = (
        df['sentiment_label']
        .value_counts()
        .reindex(['Positive', 'Neutral', 'Negative'])
        .fillna(0)
        .astype(int)
    )
    topic_counts = df['topic_id'].value_counts().sort_index()
    rating_col = next((candidate for candidate in ['rating', 'stars', 'score'] if candidate in df.columns), None)

    summary = generate_summary(
        sentiment_counts=sentiment_counts.to_dict(),
        topic_keywords=topic_model_info['topic_keywords'],
        sample_reviews=df[review_col].head(20).tolist(),
    )

    pie_fig = px.pie(
        names=sentiment_counts.index.tolist(),
        values=sentiment_counts.values.tolist(),
        title='Sentiment distribution',
        color=sentiment_counts.index.tolist(),
        color_discrete_map={'Positive': '#16a34a', 'Neutral': '#64748b', 'Negative': '#dc2626'},
    )

    bar_fig = px.bar(
        x=[f'Topic {int(i)}' for i in topic_counts.index.tolist()],
        y=topic_counts.values.tolist(),
        title='Topics by review count',
        labels={'x': 'Topic', 'y': 'Reviews'},
        color=topic_counts.values.tolist(),
        color_continuous_scale='Blues',
    )

    rating_html = None
    if rating_col:
        rating_fig = px.histogram(df, x=rating_col, title=f'{rating_col.title()} distribution')
        rating_html = rating_fig.to_html(full_html=False, include_plotlyjs='cdn')

    wordcloud_b64 = _build_wordcloud_b64(df[review_col])
    out_file = UPLOAD_DIR / f'processed_{file_path.stem}.csv'
    df.to_csv(out_file, index=False)

    state = {
        'source_file': file_path.name,
        'processed_file': out_file.name,
        'review_column': review_col,
        'rows': _df_to_records(df),
        'summary': summary,
        'stats': {
            'total_reviews': int(len(df)),
            'positive_reviews': int(sentiment_counts.get('Positive', 0)),
            'neutral_reviews': int(sentiment_counts.get('Neutral', 0)),
            'negative_reviews': int(sentiment_counts.get('Negative', 0)),
            'topic_count': int(df['topic_id'].nunique()),
        },
        'charts': {
            'sentiment_html': pie_fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'topics_html': bar_fig.to_html(full_html=False, include_plotlyjs=False),
            'rating_html': rating_html,
        },
        'wordcloud': wordcloud_b64,
        'topics': topic_model_info['topic_keywords'],
    }
    _save_state(state)
    return state


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part found in the request.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if not file or file.filename == '':
        flash('Please choose a CSV file.', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Only CSV files are supported.', 'error')
        return redirect(url_for('index'))

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name
    file.save(file_path)

    try:
        payload = process_reviews(file_path)
    except Exception as exc:
        flash(str(exc), 'error')
        return redirect(url_for('index'))

    session['processed_file'] = payload['processed_file']
    flash('File processed successfully.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    state = _load_state()
    return render_template('dashboard.html', state=state)


@app.route('/download')
def download():
    processed_file = session.get('processed_file')
    if not processed_file:
        flash('Nothing to download yet.', 'error')
        return redirect(url_for('index'))
    path = UPLOAD_DIR / processed_file
    return send_file(path, as_attachment=True, download_name=path.name)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'app': 'ReviewSenseAI', 'version': '1.0.0'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
