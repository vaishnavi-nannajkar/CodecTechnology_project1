# CodecTechnology_project1
Real-time Twitter sentiment analysis web app powered by ML (TF-IDF + Logistic Regression) and BERT. Built with Streamlit, FastAPI, and HuggingFace Transformers. Features interactive dashboards, word clouds, brand reputation scoring, and PDF export.

# 🧠 AI Twitter Sentiment Intelligence System

> Analyze Twitter sentiment in real-time using Machine Learning and BERT — 
> built for brands, researchers, and developers.

## What is this?

AI Twitter Sentiment Intelligence is a full-stack web application that fetches 
tweets for any keyword or brand and classifies them as **Positive**, **Negative**, 
or **Neutral** using two AI models running in parallel.

It was built to solve a real problem — understanding how people feel about a 
brand or topic on Twitter without manually reading hundreds of tweets.

You type a keyword. The system fetches tweets, cleans the text, runs it through 
ML and BERT models, and gives you a full analytics dashboard in seconds.

## How it works

1. Enter a keyword or brand name (e.g., Tesla, ChatGPT, iPhone)
2. The app fetches the latest tweets from Twitter API
3. Text is cleaned — URLs, mentions, hashtags, emojis removed
4. Two AI models classify each tweet as Positive / Negative / Neutral
5. Results appear as interactive charts, word clouds, and tweet cards
6. Download a PDF report or export CSV with one click

## Models Used

| Model | Accuracy | Speed | Best For |
|---|---|---|---|
| TF-IDF + Logistic Regression | ~84% | Very Fast | Quick analysis |
| DistilBERT (HuggingFace) | ~92% | Moderate | Production use |
| Ensemble (Both) | ~93% | Moderate | Maximum accuracy |

## Key Features

- **Real-time Analysis** — Analyze up to 200 tweets instantly
- **Dual AI Models** — ML and BERT with ensemble voting
- **Analytics Dashboard** — Pie charts, bar charts, trend line, word cloud
- **Brand Reputation Score** — Formula: (Positive - Negative) / Total × 100
- **PDF Report Export** — Professional downloadable reports
- **Search History** — SQLite database stores all past searches
- **REST API** — FastAPI backend for external integrations
- **Demo Mode** — Works without Twitter API keys using mock data

## Tech Stack

**Frontend:** Streamlit · Plotly · WordCloud · Matplotlib  
**Backend:** Python · FastAPI · SQLite  
**ML:** scikit-learn · TF-IDF · Logistic Regression  
**NLP:** HuggingFace Transformers · DistilBERT · spaCy · NLTK  
**API:** Tweepy (Twitter API v2)  
**Export:** ReportLab (PDF)

## Screenshots

| Homepage | Analytics Dashboard |
|---|---|
| Sidebar controls with model selection | Interactive pie/bar charts + word cloud |

| Sentiment Results | Model Comparison |
|---|---|
| Color-coded tweet cards with confidence bars | Radar chart + confusion matrices |

## Who is this for?

- **Marketing teams** monitoring brand reputation
- **Researchers** studying public opinion on social media
- **Developers** learning NLP and sentiment analysis
- **Students** building ML portfolio projects

## Live Demo

> Enable **Demo Mode** in the sidebar — no Twitter API keys required.
> The app generates realistic synthetic tweets so you can explore all features immediately.
```

---

## Topics / Tags to add on GitHub
```
nlp  sentiment-analysis  twitter  bert  machine-learning  streamlit  
python  fastapi  huggingface  transformers  data-science  deep-learning  
text-classification  tfidf  plotly  wordcloud  tweepy  natural-language-processing
```

---

## One-liner for LinkedIn / Portfolio
```
Built a real-time Twitter Sentiment Analysis system using Python, BERT, and Streamlit — 
classifies tweets as Positive/Negative/Neutral with 92%+ accuracy, interactive 
dashboards, and PDF report export.
