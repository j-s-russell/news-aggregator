import streamlit as st
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DBNAME")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

display_article_bias = False

def get_bias_color(val):
    if val <= -2:
        return "#2563EB"  # left - blue
    elif val == -1:
        return "#60A5FA"  # center-left - light blue
    elif val == 0:
        return "#9CA3AF"  # center - gray
    elif val == 1:
        return "#FCA5A5"  # center-right - light red
    elif val >= 2:
        return "#DC2626"  # right - red


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
        st.markdown(f"🔵 **Left:** {scores_vector.get('left-wing', 0):.1%}")
    with col2:
        st.markdown(f"⚪ **Center:** {scores_vector.get('center', 0):.1%}")
    with col3:
        st.markdown(f"🔴 **Right:** {scores_vector.get('right-wing', 0):.1%}")
        

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
st.set_page_config(page_title="🗞️ News Aggregator", layout="wide")

# Load articles
articles = get_articles()
articles.reset_index(drop=True, inplace=True)
articles['id'] = range(len(articles))
embedding_matrix = np.stack(articles['embedding'].values)

query_params = st.query_params
selected_id = st.query_params.get("article_id")

if selected_id:
    try:
        selected_id = int(selected_id)
    except (ValueError, TypeError):
        selected_id = None


# ----------------- DETAIL PAGE -----------------
if selected_id is not None:
    selected_id = int(selected_id)
    article = articles[articles['id'] == selected_id].iloc[0]
    idx = articles[articles['id'] == selected_id].index[0]

    st.title(article['title'])
    st.write(f"*{article['source']}* • {article['publish_date'].strftime('%B %d, %Y %H:%M')} • **Topic:** {article['cluster_label']}")
    if article.get("author"):
        st.caption(f"By {article['author']}")

    # Bias bar
    source_bias_val = article['source_bias']
    marker_color = get_bias_color(source_bias_val)

    st.markdown(
        f"""
        <div style='height: 16px; background: linear-gradient(to right, 
             #2563EB 0%, 
             #60A5FA 25%, 
             #9CA3AF 50%, 
             #FCA5A5 75%, 
             #DC2626 100%);
             position: relative; border-radius: 6px; margin-bottom: 16px; margin-top: 8px;'>
            <div style='
                position: absolute;
                left: {50 + source_bias_val * 25}%;
                transform: translateX(-50%);
                width: 16px;
                height: 32px;
                background: {marker_color};
                border: 2px solid #111;
                border-radius: 4px;
                box-shadow: 0 0 8px rgba(0,0,0,0.4);
                z-index: 2;
            '></div>
        </div>
        <div style='text-align: center; font-size: 0.85em; margin-top: -8px; margin-bottom: 16px; color: #FFFFFF;'>
            <strong>Source Bias</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if display_article_bias:
        if article['article_bias'] is None:
            st.info("**Estimated Article Bias:** Not Applicable (Non-Political Content)")
            
        else:
            scores = {
                'left-wing': article['article_bias'][0],
                'center': article['article_bias'][1], 
                'right-wing': article['article_bias'][2]
            }
            st.markdown("**Estimated Article Bias:** 🔵 Left: {:.1%} | ⚪ Center: {:.1%} | 🔴 Right: {:.1%}".format(
                scores.get('left-wing', 0),
                scores.get('center', 0), 
                scores.get('right-wing', 0)
            ))

    st.subheader("Summary")
    summary_type = st.radio(
        "Choose summary type:",
        options=["Abstractive", "Extractive"],
        horizontal=True,
        key=f"summary_type_{selected_id}"
    )
    
    # Display the selected summary
    if summary_type == "Abstractive":
        st.write(article['abs_summary'])
    elif summary_type == "Extractive":
        st.write(article['ext_summary'])
    st.markdown(f"[Read original article]({article['url']})", unsafe_allow_html=True)

    st.subheader("Similar Articles")
    similar_articles = get_similar_articles(embedding_matrix, idx, top_n=5)
    for _, sim_row in similar_articles.iterrows():
        sim_id = sim_row['id']
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{sim_row['title']}** – {sim_row['source']}")
        with col2:
            if st.button("View", key=f"similar_{sim_id}"):
                st.query_params["article_id"] = str(sim_id)
                st.rerun()

    if st.button("⬅ Back to All Articles"):
        st.query_params.clear()
        st.rerun()

    st.stop()


# ----------------- MAIN PAGE -----------------
st.title("News Aggregator")
st.write("Browse articles from various sources")

sources = st.multiselect("Filter by source", options=articles['source'].unique(), default=articles['source'].unique())
cluster_labels = st.multiselect("Filter by topic", options=sorted(articles['cluster_label'].dropna().unique()), default=sorted(articles['cluster_label'].dropna().unique()))
keywords = st.text_input("Search in title or description")

filtered = articles[
    articles['source'].isin(sources) & 
    articles['cluster_label'].isin(cluster_labels)
]
if keywords:
    keyword_lower = keywords.lower()
    filtered = filtered[
        filtered['title'].str.lower().str.contains(keyword_lower) |
        filtered['description'].str.lower().str.contains(keyword_lower)
    ]

# Separate top articles from regular articles
top_articles = filtered[filtered['top'] == True]
regular_articles = filtered[filtered['top'] == False]

def display_article(row):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### {row['title']}")
    with col2:
        if st.button("Read More", key=f"main_{row['id']}"):
            st.query_params["article_id"] = str(row['id'])
            st.rerun()
            
    st.write(f"*{row['source']}* • {row['publish_date'].strftime('%B %d, %Y %H:%M')} • **Topic:** {row['cluster_label']}")
    if row.get("author"):
        st.caption(f"By {row['author']}")
    
    # Bias bar
    source_bias_val = row['source_bias']
    marker_color = get_bias_color(source_bias_val)
    st.markdown(
        f"""
        <div style='height: 16px; background: linear-gradient(to right, 
             #2563EB 0%, 
             #60A5FA 25%, 
             #9CA3AF 50%, 
             #FCA5A5 75%, 
             #DC2626 100%);
             position: relative; border-radius: 6px; margin-bottom: 16px; margin-top: 8px;'>
            <div style='
                position: absolute;
                left: {50 + source_bias_val * 25}%;
                transform: translateX(-50%);
                width: 16px;
                height: 32px;
                background: {marker_color};
                border: 2px solid #111;
                border-radius: 4px;
                box-shadow: 0 0 8px rgba(0,0,0,0.4);
                z-index: 2;
            '></div>
        </div>
        <div style='text-align: center; font-size: 0.85em; margin-top: -8px; margin-bottom: 16px; color: #FFFFFF;'>
            <strong>Source Bias</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

# Display top articles section
if not top_articles.empty:
    st.markdown("## Top Headlines")
    st.markdown("*Featured stories of the day*")
    
    with st.container(height=600, border=True):
        for idx, row in top_articles.iterrows():
            display_article(row)

# Display regular articles section
if not regular_articles.empty:
    st.markdown("## All Articles")
    if not top_articles.empty:  # Only show this subtitle if there are top articles above
        st.markdown("*Browse all other stories*")
    
    with st.container(height=800, border=True):
        for idx, row in regular_articles.iterrows():
            display_article(row)
