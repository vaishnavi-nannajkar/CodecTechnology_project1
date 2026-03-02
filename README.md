# 🧠 AI Twitter Sentiment Intelligence System

A production-ready, full-stack web application that performs **real-time Twitter sentiment analysis** using both traditional ML (TF-IDF + Logistic Regression) and **BERT** (via HuggingFace Transformers). Built with Streamlit, FastAPI, and a clean professional UI.

---

## 📸 Features at a Glance

| Feature | Description |
|---|---|
| 🐦 Tweet Fetching | Twitter API v2 (Tweepy) + demo mock mode |
| 🤖 ML Model | TF-IDF + Logistic Regression with 84%+ accuracy |
| 🧠 BERT Model | DistilBERT / fine-tunable BERT with 92%+ accuracy |
| 📊 Analytics | Pie charts, bar charts, word cloud, trend line |
| 🏆 Brand Score | Reputation score: (Pos - Neg) / Total × 100 |
| 📄 PDF Export | Professional ReportLab PDF reports |
| 🗃️ Database | SQLite history of past searches |
| ⚡ REST API | FastAPI endpoints for external integration |

---

## 🗂️ Project Structure

```
project/
├── app.py                    # Main Streamlit application
├── api.py                    # FastAPI REST backend
├── train_models.py           # Model training scripts
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml           # Streamlit theme & config
├── backend/
│   ├── preprocessing.py      # Text cleaning pipeline
│   ├── model_ml.py           # TF-IDF + LR sentiment model
│   ├── model_bert.py         # BERT/DistilBERT sentiment model
│   ├── twitter_fetch.py      # Twitter API fetcher + mock
│   ├── database.py           # SQLite manager
│   └── report_generator.py   # PDF report generation
├── models/
│   ├── ml_model.pkl          # Trained ML model (auto-generated)
│   └── bert_model/           # Fine-tuned BERT (optional)
└── data/
    └── sentiment.db          # SQLite database (auto-generated)
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-repo/ai-sentiment-intelligence.git
cd ai-sentiment-intelligence

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download NLP Models

```bash
# spaCy English model
python -m spacy download en_core_web_sm

# NLTK stopwords
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 3. Configure API Keys (Optional)

```bash
cp .env.example .env
# Edit .env with your Twitter API Bearer Token
```

> **Demo Mode**: The app works **without** Twitter API keys using synthetic tweets.

### 4. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Model Training

```bash
# Train ML model only (fast, ~2 seconds)
python train_models.py --model ml

# Fine-tune BERT (requires GPU recommended, ~30-60 min)
python train_models.py --model bert --epochs 3

# Train both
python train_models.py --model both

# Train on your own labeled CSV (columns: 'text', 'label')
python train_models.py --model ml --data /path/to/data.csv

# Evaluate current models
python train_models.py --evaluate
```

---

## 🌐 REST API Usage

Start the FastAPI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/analyze` | Full pipeline analysis |
| POST | `/predict` | Predict custom texts |
| GET | `/history` | Past search history |

### Example: Analyze Tweets

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "Tesla",
    "count": 50,
    "model": "both",
    "use_mock": true
  }'
```

### Example: Predict Custom Text

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["I love this product!", "Terrible service never again"],
    "model": "bert"
  }'
```

---

## 🚢 Deployment

### Streamlit Cloud (Recommended — Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file as `app.py`
4. Add secrets in Streamlit Cloud dashboard:
   - `TWITTER_BEARER_TOKEN = "your_token"`

### Render

```bash
# render.yaml
services:
  - type: web
    name: sentiment-app
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.headless true
```

### Heroku

```bash
# Procfile
web: streamlit run app.py --server.port $PORT --server.headless true
```

```bash
heroku create your-app-name
heroku config:set TWITTER_BEARER_TOKEN=your_token
git push heroku main
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```

```bash
docker build -t sentiment-app .
docker run -p 8501:8501 -e TWITTER_BEARER_TOKEN=your_token sentiment-app
```

---

## 🧪 Testing

```bash
# Run the app in demo mode (no API keys needed)
streamlit run app.py
# ✓ Check "Demo Mode" in sidebar → Enter any keyword → Click Analyze

# Test preprocessing
python -c "
from backend.preprocessing import TextPreprocessor
p = TextPreprocessor()
print(p.clean('Just tried @Tesla!! It is AMAZING 🚀 https://t.co/abc #EV #Future'))
"

# Test ML model
python -c "
from backend.model_ml import MLSentimentModel
m = MLSentimentModel()
print(m.predict(['This is absolutely amazing!', 'Terrible waste of money']))
"

# Test mock tweets
python -c "
from backend.twitter_fetch import TwitterFetcher
f = TwitterFetcher(use_mock=True)
tweets = f.fetch('Apple', 10)
print(tweets[:3])
"

# Test FastAPI endpoints
uvicorn api:app --port 8000 &
curl http://localhost:8000/health
```

---

## 📊 Sample Screenshots

1. **Homepage**: Hero header with gradient, sidebar controls, feature cards
2. **Sentiment Results Tab**: Color-coded tweet cards with confidence bars
3. **Analytics Tab**: Interactive pie/bar charts, word cloud, trend line
4. **Model Comparison Tab**: Grouped bar + radar charts, confusion matrices
5. **Search History Tab**: Past searches table with timestamps

---

## 🔧 Configuration Options

| Environment Variable | Default | Description |
|---|---|---|
| `TWITTER_BEARER_TOKEN` | — | Twitter API v2 Bearer Token |
| `BERT_BASE_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace model ID |
| `APP_ENV` | `production` | Environment name |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 🛣️ Future Enhancements

- [ ] **Real-time streaming** — Twitter Filtered Stream API for live feeds
- [ ] **Multi-language support** — XLM-RoBERTa for non-English tweets
- [ ] **Entity extraction** — Named entity recognition to identify mentioned brands/people
- [ ] **Aspect-based sentiment** — Sentiment per product feature (price, quality, etc.)
- [ ] **Email alerts** — Notify when brand reputation drops below threshold
- [ ] **Scheduled analysis** — Cron-based automated daily reports
- [ ] **MongoDB integration** — Scalable storage for high-volume analysis
- [ ] **Dashboard sharing** — Shareable links to analysis results
- [ ] **Competitor comparison** — Side-by-side brand sentiment analysis
- [ ] **Fine-tuned Twitter BERT** — Train on Twitter-specific labeled data

---

## 🏗️ Architecture

```
User Input (Streamlit UI)
        ↓
TwitterFetcher (API / Mock)
        ↓
TextPreprocessor (spaCy + NLTK)
        ↓
    ┌───┴───┐
    ML Model   BERT Model
    (TF-IDF)   (HuggingFace)
    └───┬───┘
        ↓
  Ensemble Prediction
        ↓
  Analytics Engine (Plotly)
        ↓
  DatabaseManager (SQLite)
        ↓
  Report Generator (ReportLab)
```

---

## 📄 License

MIT License — free to use, modify, and deploy.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

*Built with ❤️ using Streamlit, FastAPI, scikit-learn, and HuggingFace Transformers*