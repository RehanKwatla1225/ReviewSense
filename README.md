# ReviewSense AI

Flask-based ReviewSense AI app for CSV review analysis.

## What it does
- Upload a CSV file of reviews
- Detect and clean the review text column
- Run sentiment analysis using Hugging Face Transformers
- Cluster reviews into topics using LDA
- Generate an AI executive summary with Groq, or a fallback summary when the API key is missing
- Show charts, a word cloud, and a processed table in the dashboard

## Run locally

```bash
cd reviewsense
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env .env.local  # optional
python app.py
```

Open http://localhost:5000

## Notes
- Put your Groq API key in `GROQ_API_KEY`
- The app also works without the key using the fallback summary
- Sample data lives in `sample_data/amazon_reviews.csv`
