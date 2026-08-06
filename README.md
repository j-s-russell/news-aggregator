# 🗞️ News Aggregator

A news aggregation and analysis platform that fetches articles from multiple sources, summarizes and clusters them, and presents them through a dark-themed Streamlit web interface — complete with a retrieval-augmented generation (RAG) chat for asking questions about the news.

## ✨ Features

- **Multi-source Article Fetching**: Pulls articles from NewsAPI with configurable sources
- **Content Extraction**: Scrapes full article content from URLs with `newspaper3k`
- **Dual Summarization**: Abstractive summaries (Gemini) + extractive summaries (LSA via `sumy`)
- **Topic Clustering**: Groups articles with KMeans (or HDBSCAN) and labels clusters via Gemini
- **Cluster Summaries**: A 4–5 sentence overview generated for each topic during the pipeline and cached in the database
- **Similar Articles**: Cosine-similarity matching on article embeddings
- **RAG Chat**: A multi-turn sidebar chat that answers questions about the news using pgvector retrieval and DeepSeek
- **Dark-Themed UI**: Three-page navigation (main → cluster → detail) with topic browsing and bias indicators
- **Automated Pipeline**: Runs on a schedule via GitHub Actions

## Architecture

```
news-aggregator/
├── main.py                    # Pipeline orchestration (fetch → process → cluster)
├── streamlit_app.py           # Streamlit web application
├── config.py                  # Environment/config loading
├── retrievers/
│   └── fetcher.py             # Article fetching from NewsAPI + content extraction
├── processors/
│   ├── summarizer.py          # Abstractive (Gemini) & extractive (LSA) summarization
│   ├── clusterer.py           # Topic clustering, labeling, and cluster summaries
│   ├── chunker.py             # Article chunking + embedding for RAG
│   ├── rag.py                 # RAG retrieval + DeepSeek generation
│   ├── cluster_tuning.py      # Standalone clustering hyperparameter tuning
│   └── bias_classifier.py     # Bias classification (experimental, not in pipeline)
└── database/
    └── db_client.py           # Database operations and connections
```

## Pipeline

`main.py` orchestrates the full processing flow:

1. **Fetch** — pull raw articles from NewsAPI
2. **Summarize** — generate abstractive (Gemini) and extractive (LSA) summaries
3. **Insert** — write processed articles to `news_pipeline`
4. **Cluster** — embed articles (`all-MiniLM-L6-v2`), cluster with KMeans/HDBSCAN, label clusters via Gemini
5. **Summarize clusters** — generate a cached overview per cluster label
6. **Chunk** — split each article's extractive summary into overlapping chunks, embed them, and store in `article_chunks`

## RAG Chat

The sidebar chat (`Ask the News`) answers general news questions across the entire corpus:

1. User's question is embedded with `all-MiniLM-L6-v2`
2. The top matching chunks are retrieved via the `match_chunks` pgvector RPC
3. Chunks + conversation history are sent to DeepSeek (`deepseek-chat`)
4. The grounded answer is rendered in the chat thread

## Tech Stack

- **Language**: Python 3.11
- **UI**: Streamlit
- **Database**: Supabase PostgreSQL + pgvector
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim) via sentence-transformers
- **LLMs**: Gemini 2.5 Flash (summarization, clustering) · DeepSeek `deepseek-chat` (RAG)
- **Clustering**: scikit-learn, HDBSCAN
- **CI/CD**: GitHub Actions

## Setup

### 1. Environment variables

Create a `.env` file (or configure secrets) with:

```
USER=...
PASSWORD=...
HOST=...
PORT=...
DBNAME=...
NEWS_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Database setup

Run the following in the Supabase SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cluster_summaries (
    cluster_label text PRIMARY KEY,
    summary_text text NOT NULL,
    generated_at timestamptz DEFAULT now()
);

CREATE TABLE article_chunks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  article_id bigint REFERENCES news_pipeline(id) ON DELETE CASCADE,
  chunk_text text NOT NULL,
  chunk_index int NOT NULL,
  embedding vector(384),
  created_at timestamptz DEFAULT now()
);

CREATE INDEX ON article_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),
  match_count int default 5
)
RETURNS TABLE (
  chunk_text text,
  article_id bigint,
  title text,
  source text,
  url text,
  publish_date timestamptz,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    c.chunk_text,
    a.id AS article_id,
    a.title,
    a.source,
    a.url,
    a.publish_date,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM article_chunks c
  JOIN news_pipeline a ON a.id = c.article_id
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;
```

## Running

### Run the pipeline

```bash
python main.py
```

Flags:
- `--fetch-only` — fetch articles without processing
- `--process-only` — process existing fetched articles

### Run the web app

```bash
streamlit run streamlit_app.py
```

## GitHub Actions

The `.github/workflows/update.yml` workflow runs `main.py` on a schedule (or manually). The pipeline and app require the `USER`, `PASSWORD`, `HOST`, `DBNAME`, `NEWS_API_KEY`, and `GOOGLE_API_KEY` secrets in GitHub.
