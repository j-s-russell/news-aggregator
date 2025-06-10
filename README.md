# 🗞️ News Aggregator

A comprehensive news aggregation and analysis platform that fetches articles from multiple sources, processes them with advanced NLP techniques, and presents them through an intuitive Streamlit web interface.

## ✨ Features

- **Multi-source Article Fetching**: Pulls articles from NewsAPI with configurable sources
- **Content Extraction**: Scrapes full article content from URLs
- **Dual Summarization**: Generates both abstractive and extractive summaries using LSA and LLMs
- **Bias Classification**: Automatically classifies source bias to help users understand perspective
- **Topic Clustering**: Groups and labels related articles using KMeans and Gemini
- **Interactive Web Interface**: Clean, responsive Streamlit dashboard for browsing articles
- **Database Integration**: Stores processed articles in Supabase PostgreSQL database
- **Smart Filtering**: Filter by source, topic, keywords, and publication date

## Architecture

```
news-aggregator/
├── main.py                    # Main orchestration script
├── app.py                     # Streamlit web application
├── retrievers/
│   └── fetcher.py            # Article fetching from NewsAPI
├── processors/
│   ├── summarizer.py         # Text summarization (abstractive & extractive)
│   ├── clusterer.py          # Topic clustering
│   └── bias_classifier.py   # Source bias classification
└── database/
    └── db_client.py          # Database operations and connections
```
