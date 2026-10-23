import streamlit as st
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from database.db_client import get_cluster_summary
from ui.components import (
    inject_css,
    bias_bar_html,
    meta_html,
    author_html,
    article_card_html,
    similar_row_html,
    page_header_html,
    section_label,
)

load_dotenv(override=True)

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DBNAME")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

display_article_bias = False


def get_similar_articles(embedding_matrix, selected_index, top_n=3):
    selected_vector = embedding_matrix[selected_index].reshape(1, -1)
    similarities = cosine_similarity(selected_vector, embedding_matrix)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]  # skip self
    return articles.iloc[similar_indices][['id', 'title', 'url', 'abs_summary', 'ext_summary', 'source']]


def display_bias_scores(scores_vector, threshold=0.7):
    # Check if non-political
    if scores_vector.get('non-political', 0) > threshold:
        st.info("**Political Bias:** Not Applicable (Non-political content)")
        return

    # Display scores in columns
    st.markdown("#### Political Bias Scores")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Left:** {scores_vector.get('left-wing', 0):.1%}")
    with col2:
        st.markdown(f"**Center:** {scores_vector.get('center', 0):.1%}")
    with col3:
        st.markdown(f"**Right:** {scores_vector.get('right-wing', 0):.1%}")


# ---------- CONNECT TO DATABASE ----------
@st.cache_data(ttl=600)
def get_articles():
    conn = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    query = "SELECT * FROM news_pipeline ORDER BY publish_date DESC;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="News Aggregator", layout="wide")
inject_css()


# Load articles
articles = get_articles()
articles.reset_index(drop=True, inplace=True)
articles['id'] = range(len(articles))

max_len = max(len(e) for e in articles['embedding'] if e is not None)

# Pad/truncate function
def pad_embedding(e):
    if e is None:
        return np.zeros(max_len)
    e = np.array(e)  # convert list to numpy array
    if len(e) < max_len:
        return np.pad(e, (0, max_len - len(e)))
    return e[:max_len]

# Apply to all embeddings
embedding_matrix = np.stack(articles['embedding'].apply(pad_embedding).values)

# Use session_state as the sole source of truth for navigation
if "article_id" not in st.session_state:
    st.session_state["article_id"] = None
if "cluster_page" not in st.session_state:
    st.session_state["cluster_page"] = None
if "return_to" not in st.session_state:
    st.session_state["return_to"] = "main"

selected_id = st.session_state["article_id"]
cluster_page = st.session_state["cluster_page"]


# ----------------- SIDEBAR CHAT -----------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

with st.sidebar:
    st.html(
        '<div class="sidebar-head"><h3>Ask the News</h3>'
        '<p>Questions answered from today\'s articles.</p></div>'
    )

    question = st.chat_input("Ask a question...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching articles..."):
                from processors.rag import answer_question
                answer = answer_question(
                    question,
                    chat_history=st.session_state["chat_history"],
                    match_count=15,
                )
            st.write(answer)

        st.session_state["chat_history"].append({"role": "user", "content": question})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state["chat_history"]:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()


# ----------------- DETAIL PAGE -----------------
if selected_id is not None:
    selected_id = int(selected_id)
    article = articles[articles['id'] == selected_id].iloc[0]
    idx = articles[articles['id'] == selected_id].index[0]

    # Back button
    back_label = f"← {cluster_page}" if st.session_state.get("return_to") == "cluster" else "← All articles"
    if st.button(back_label, key="back"):
        st.session_state["article_id"] = None
        if st.session_state.get("return_to") != "cluster":
            st.session_state["cluster_page"] = None
        st.rerun()

    st.html(page_header_html(article['title']))

    st.html(meta_html(article) + author_html(article) + bias_bar_html(article['source_bias'], show_scale=True))

    if display_article_bias:
        if article['article_bias'] is None:
            st.info("**Estimated Article Bias:** Not Applicable (Non-Political Content)")
        else:
            scores = {
                'left-wing': article['article_bias'][0],
                'center': article['article_bias'][1],
                'right-wing': article['article_bias'][2]
            }
            st.markdown("**Estimated Article Bias:** Left: {:.1%} | Center: {:.1%} | Right: {:.1%}".format(
                scores.get('left-wing', 0),
                scores.get('center', 0),
                scores.get('right-wing', 0)
            ))

    section_label("Summary")
    with st.container(key="summary"):
        summary_type = st.segmented_control(
            "Summary type",
            options=["Abstractive", "Extractive"],
            default="Abstractive",
            required=True,
            label_visibility="collapsed",
            key=f"summary_type_{selected_id}"
        )

        if summary_type == "Abstractive":
            st.write(article['abs_summary'])
        elif summary_type == "Extractive":
            st.write(article['ext_summary'])

        st.link_button("Read original ↗", article['url'])

    section_label("Similar articles")
    similar_articles = get_similar_articles(embedding_matrix, idx, top_n=5)
    for _, sim_row in similar_articles.iterrows():
        sim_id = sim_row['id']
        with st.container(key=f"sim_{sim_id}"):
            col_text, col_btn = st.columns([5, 1], vertical_alignment="center")
            with col_text:
                st.html(similar_row_html(sim_row))
            with col_btn:
                if st.button("Open →", key=f"similar_{sim_id}"):
                    st.session_state["article_id"] = int(sim_id)
                    st.session_state["return_to"] = "cluster" if st.session_state.get("cluster_page") else "main"
                    st.rerun()

    st.stop()


# ----------------- CLUSTER PAGE -----------------
if cluster_page is not None:
    cluster_articles = articles[articles['cluster_label'] == cluster_page]

    # Back button
    if st.button("← All articles", key="back"):
        st.session_state["cluster_page"] = None
        st.rerun()

    st.html(page_header_html(
        cluster_page,
        f'{len(cluster_articles)} article{"s" if len(cluster_articles) != 1 else ""}',
        eyebrow="Topic",
    ))

    # ---------- CLUSTER SUMMARY ----------
    cached = get_cluster_summary(cluster_page)
    if cached:
        with st.container(key="summary"):
            st.html('<div class="eyebrow">AI summary</div>')
            st.write(cached['summary_text'])

    # ---------- ARTICLES IN CLUSTER ----------
    section_label("Articles")
    for _, row in cluster_articles.iterrows():
        with st.container(key=f"card_{row['id']}"):
            st.html(article_card_html(row, show_topic=False))
            if st.button("Read →", key=f"cluster_{row['id']}"):
                st.session_state["article_id"] = int(row['id'])
                st.session_state["return_to"] = "cluster"
                st.rerun()

    st.stop()


# ----------------- MAIN PAGE -----------------
latest = articles['publish_date'].max()
st.html(page_header_html(
    "News",
    f"{len(articles)} articles · updated {latest.strftime('%b %d, %H:%M')}" if pd.notna(latest) else "",
))

all_cluster_labels = sorted(articles['cluster_label'].dropna().unique())

# --- Toolbar: search, view mode, filters ---
with st.container(key="filters"):
    col_search, col_view, col_filters = st.columns([5, 4, 1.4], vertical_alignment="center")
    with col_search:
        keywords = st.text_input(
            "Search",
            placeholder="Search titles and descriptions",
            label_visibility="collapsed",
        )
    with col_view:
        view_mode = st.segmented_control(
            "View",
            options=["Top Headlines", "All Articles"],
            default="Top Headlines",
            required=True,
            label_visibility="collapsed",
        )

    with col_filters, st.popover("Filters", width="stretch"):
        sources = st.multiselect("Sources", options=articles['source'].unique(), default=articles['source'].unique())
        active_clusters = st.pills(
            "Topics",
            options=all_cluster_labels,
            selection_mode="multi",
            default=all_cluster_labels,
            key="topic_filter",
        ) or []

# --- Browse by topic ---
section_label("Browse by topic")
with st.container(horizontal=True, key="topics"):
    for label in all_cluster_labels:
        if st.button(label, key=f"go_{label}"):
            st.session_state["cluster_page"] = label
            st.rerun()

# --- Filter + display ---
filtered = articles[
    articles['source'].isin(sources) &
    articles['cluster_label'].isin(active_clusters)
]
if keywords:
    keyword_lower = keywords.lower()
    filtered = filtered[
        filtered['title'].str.lower().str.contains(keyword_lower) |
        filtered['description'].str.lower().str.contains(keyword_lower)
    ]

if view_mode == "Top Headlines":
    display_df = filtered[filtered['top'] == True]
else:
    display_df = filtered


def display_article(row):
    with st.container(key=f"card_{row['id']}"):
        st.html(article_card_html(row))
        if st.button("Read →", key=f"main_{row['id']}"):
            st.session_state["article_id"] = int(row['id'])
            st.session_state["return_to"] = "main"
            st.rerun()


section_label(view_mode)
if not display_df.empty:
    for _, row in display_df.iterrows():
        display_article(row)
else:
    with st.container(key="empty"):
        st.write("No articles match the current filters.")
