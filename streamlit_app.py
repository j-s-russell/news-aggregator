import streamlit as st
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from database.db_client import get_cluster_summary

load_dotenv(override=True)

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

# ---------- GLOBAL STYLES ----------
st.markdown("""
<style>
    /* Page background */
    .stApp {
        background-color: #0F172A;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #F1F5F9 !important;
    }

    /* Body text */
    p, span, div {
        color: #CBD5E1;
    }

    /* Section headers */
    h2 {
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
        margin-bottom: 4px;
        margin-top: 28px;
    }
    h3 {
        margin-top: 24px;
    }

    /* Article card container */
    .article-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 0;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .article-card:hover {
        border-color: #475569;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* Card title */
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F1F5F9 !important;
        margin: 0 0 8px 0 !important;
        line-height: 1.4;
    }

    /* Card meta row */
    .card-meta {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .card-source {
        font-weight: 600;
        color: #60A5FA;
    }
    .card-topic {
        background: #334155;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        color: #94A3B8;
    }
    .card-author {
        font-size: 0.8rem;
        color: #64748B;
        display: block;
        margin-bottom: 8px;
    }

    /* Bias bar */
    .bias-bar-wrap {
        margin-top: 10px;
    }
    .bias-gradient {
        height: 6px;
        background: linear-gradient(to right,
            #93B4D4 0%,
            #B0C4D8 25%,
            #C8C8C8 50%,
            #D8B0B0 75%,
            #D49494 100%);
        border-radius: 3px;
        position: relative;
    }
    .bias-marker {
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 12px;
        height: 12px;
        border-radius: 50%;
        border: 2px solid #1E293B;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
    }
    .bias-label {
        font-size: 0.7rem;
        color: #64748B;
        text-align: center;
        margin-top: 4px;
    }

    /* Full-width "Read more" button that blends with card */
    .card-btn-wrap .stButton {
        padding: 0 !important;
        margin: 0 !important;
    }
    .card-btn-wrap .stButton > button {
        width: 100%;
        background: transparent !important;
        border: none !important;
        border-top: 1px solid #334155 !important;
        color: #60A5FA !important;
        font-size: 0.85rem !important;
        padding: 10px 0 4px 0 !important;
        margin: 0 !important;
        transition: background 0.15s ease !important;
        border-radius: 0 0 10px 10px !important;
        text-align: right !important;
    }
    .card-btn-wrap .stButton > button:hover {
        background: #253349 !important;
        color: #93C5FD !important;
    }

    /* Detail page back button */
    .back-btn .stButton > button {
        background: #1E293B !important;
        color: #CBD5E1 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        transition: background 0.15s ease !important;
    }
    .back-btn .stButton > button:hover {
        background: #334155 !important;
        color: #F1F5F9 !important;
    }

    /* Similar article buttons */
    .similar-btn .stButton > button {
        background: #1E293B !important;
        color: #60A5FA !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        padding: 4px 12px !important;
        transition: background 0.15s ease !important;
    }
    .similar-btn .stButton > button:hover {
        background: #334155 !important;
    }

    /* Filter bar */
    .filter-bar {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
    }

    /* Filter inputs — dark theme overrides */
    [data-testid="stMultiSelect"] [data-baseweb="select"],
    [data-testid="stTextInput"] [data-baseweb="input"] {
        background-color: #0F172A !important;
        border-color: #334155 !important;
        color: #CBD5E1 !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: #334155 !important;
        color: #CBD5E1 !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="select"] span,
    [data-testid="stTextInput"] span {
        color: #CBD5E1 !important;
    }
    [data-baseweb="input"] {
        color: #CBD5E1 !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="input"] span {
        color: #CBD5E1 !important;
    }
    .stMultiSelect [role="listbox"],
    .stMultiSelect [data-baseweb="menu"] {
        background-color: #1E293B !important;
    }
    [role="option"], [role="listbox"] li {
        color: #CBD5E1 !important;
    }

    /* Radio buttons */
    .stRadio > div {
        gap: 12px;
    }

    /* Scrollable containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #334155 !important;
        border-radius: 12px !important;
        background: #0F172A;
    }

    /* Horizontal rule */
    hr {
        border: none;
        border-top: 1px solid #334155;
        margin: 0;
    }

    /* Divider between sections */
    .section-divider {
        border: none;
        border-top: 1px solid #334155;
        margin: 4px 0 16px 0;
    }

    /* Similar articles cards */
    .similar-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .similar-card:hover {
        border-color: #475569;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .similar-card .sim-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #F1F5F9;
        margin: 0 0 4px 0;
    }
    .similar-card .sim-source {
        font-size: 0.8rem;
        color: #94A3B8;
    }
    .similar-card .sim-source span {
        font-weight: 600;
        color: #60A5FA;
    }

    /* Read original article link */
    .stMarkdown a {
        color: #60A5FA !important;
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: border-color 0.15s ease;
    }
    .stMarkdown a:hover {
        border-bottom-color: #60A5FA;
    }

    /* Main page header */
    .main-header {
        margin-bottom: 20px;
        padding-bottom: 16px;
    }
    .main-header h1 {
        margin: 0 0 4px 0 !important;
        font-size: 1.8rem !important;
    }
    .main-header p {
        margin: 0;
        color: #64748B !important;
        font-size: 0.9rem;
    }

    /* Remove Streamlit default borders on markdown containers */
    [data-testid="stMarkdownContainer"] {
        border: none !important;
    }

    /* Topic nav buttons — Browse by topic */
    .topic-nav-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .topic-nav-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 20px;
    }

    /* Toggle grid section label */
    .toggle-grid-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 8px;
    }

    /* Segment control overrides */
    [data-testid="stSegmentedControl"] {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 2px !important;
    }
    [data-testid="stSegmentedControl"] button {
        border-radius: 6px !important;
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background: #334155 !important;
        color: #F1F5F9 !important;
        border-color: #475569 !important;
    }

    /* Toggle labels — dark theme */
    [data-testid="stCheckbox"] label span {
        color: #CBD5E1 !important;
    }
    [data-baseweb="toggle"] {
        background-color: #334155 !important;
    }
    [data-baseweb="toggle"] span {
        background-color: #CBD5E1 !important;
    }

    /* Global button override for dark theme */
    .stButton > button {
        background: #334155 !important;
        color: #F1F5F9 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #475569 !important;
        color: #FFFFFF !important;
        border-color: #64748B !important;
    }
    .stButton > button:active,
    .stButton > button:focus {
        background: #475569 !important;
        color: #FFFFFF !important;
    }

    /* Cluster page header */
    .cluster-header {
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid #334155;
    }
    .cluster-header h1 {
        margin: 0 0 4px 0 !important;
        font-size: 1.6rem !important;
    }
    .cluster-header p {
        margin: 0;
        color: #64748B !important;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


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


# ----------------- DETAIL PAGE -----------------
if selected_id is not None:
    selected_id = int(selected_id)
    article = articles[articles['id'] == selected_id].iloc[0]
    idx = articles[articles['id'] == selected_id].index[0]

    # Back button
    with st.container():
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        back_label = f"← Back to {cluster_page}" if st.session_state.get("return_to") == "cluster" else "← Back to All Articles"
        if st.button(back_label):
            st.session_state["article_id"] = None
            if st.session_state.get("return_to") != "cluster":
                st.session_state["cluster_page"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.title(article['title'])

    meta_parts = [f"*{article['source']}*", article['publish_date'].strftime('%B %d, %Y %H:%M')]
    if article.get("cluster_label"):
        meta_parts.append(f"**Topic:** {article['cluster_label']}")
    st.write(" · ".join(meta_parts))

    if article.get("author"):
        st.caption(f"By {article['author']}")

    # Bias bar
    source_bias_val = article['source_bias']
    marker_color = get_bias_color(source_bias_val)

    st.markdown(
        f"""
        <div class="bias-bar-wrap">
            <div class="bias-gradient">
                <div class="bias-marker" style="left: {50 + source_bias_val * 25}%; background: {marker_color};"></div>
            </div>
            <div class="bias-label"><strong>Source Bias</strong></div>
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
            st.markdown("**Estimated Article Bias:** Left: {:.1%} | Center: {:.1%} | Right: {:.1%}".format(
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

    if summary_type == "Abstractive":
        st.write(article['abs_summary'])
    elif summary_type == "Extractive":
        st.write(article['ext_summary'])

    st.markdown(f"[Read original article →]({article['url']})")

    st.subheader("Similar Articles")
    similar_articles = get_similar_articles(embedding_matrix, idx, top_n=5)
    for _, sim_row in similar_articles.iterrows():
        sim_id = sim_row['id']
        sim_html = (
            f'<div class="similar-card">'
            f'<div class="sim-title">{sim_row["title"]}</div>'
            f'<div class="sim-source"><span>{sim_row["source"]}</span></div>'
            f'</div>'
        )
        st.markdown(sim_html, unsafe_allow_html=True)
        if st.button("View article", key=f"similar_{sim_id}", use_container_width=True):
            st.session_state["article_id"] = int(sim_id)
            st.session_state["return_to"] = "cluster" if st.session_state.get("cluster_page") else "main"
            st.rerun()

    st.stop()


# ----------------- CLUSTER PAGE -----------------
if cluster_page is not None:
    cluster_articles = articles[articles['cluster_label'] == cluster_page]

    # Back button
    with st.container():
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back to All Articles"):
            st.session_state["cluster_page"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cluster-header">
        <h1>{cluster_page}</h1>
        <p>{len(cluster_articles)} article{"s" if len(cluster_articles) != 1 else ""} in this topic</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- CLUSTER SUMMARY ----------
    cached = get_cluster_summary(cluster_page)
    if cached:
        st.markdown(f"""
        <div style="background:#1E293B;border:1px solid #334155;border-radius:10px;padding:20px 24px;margin-bottom:16px;">
            <p style="color:#CBD5E1;margin:0;line-height:1.6;">{cached['summary_text']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ---------- ARTICLES IN CLUSTER ----------
    st.subheader("Articles")
    for _, row in cluster_articles.iterrows():
        source_bias_val = row['source_bias']
        marker_color = get_bias_color(source_bias_val)
        author_html = f'<span class="card-author">By {row["author"]}</span>' if row.get("author") else ""

        with st.container():
            st.markdown(f"""
            <div class="article-card">
                <div class="card-title">{row['title']}</div>
                <div class="card-meta">
                    <span class="card-source">{row['source']}</span> &middot;
                    {row['publish_date'].strftime('%B %d, %Y %H:%M')}
                </div>
                {author_html}
                <div class="bias-bar-wrap">
                    <div class="bias-gradient">
                        <div class="bias-marker" style="left: {50 + source_bias_val * 25}%; background: {marker_color};"></div>
                    </div>
                    <div class="bias-label">Source Bias</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="card-btn-wrap">', unsafe_allow_html=True)
            if st.button("Read more →", key=f"cluster_{row['id']}", use_container_width=True):
                st.session_state["article_id"] = int(row['id'])
                st.session_state["return_to"] = "cluster"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# ----------------- MAIN PAGE -----------------
st.markdown("""
<div class="main-header">
    <h1>News Aggregator</h1>
    <p>Browse articles from various sources</p>
</div>
""", unsafe_allow_html=True)

# --- Filter bar: source + search ---
st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
col_source, col_search = st.columns([1, 1])
with col_source:
    sources = st.multiselect("Source", options=articles['source'].unique(), default=articles['source'].unique())
with col_search:
    keywords = st.text_input("Search in title or description")
st.markdown('</div>', unsafe_allow_html=True)

# --- Cluster toggles ---
all_cluster_labels = sorted(articles['cluster_label'].dropna().unique())
st.markdown('<div class="toggle-grid-label">Topics</div>', unsafe_allow_html=True)
toggle_cols = st.columns(3)
active_clusters = []
for i, label in enumerate(all_cluster_labels):
    with toggle_cols[i % 3]:
        if st.toggle(label, value=True, key=f"toggle_{label}"):
            active_clusters.append(label)

# --- Browse by topic ---
st.markdown('<div class="topic-nav-label">Browse by topic</div>', unsafe_allow_html=True)
topic_btn_cols = st.columns(min(len(all_cluster_labels), 6))
for i, label in enumerate(all_cluster_labels):
    with topic_btn_cols[i % min(len(all_cluster_labels), 6)]:
        if st.button(label, key=f"go_{label}", use_container_width=True):
            st.session_state["cluster_page"] = label
            st.rerun()

# --- Segmented control: Top / All ---
view_mode = st.segmented_control(
    "View",
    options=["Top Headlines", "All Articles"],
    default="Top Headlines",
)

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
    source_bias_val = row['source_bias']
    marker_color = get_bias_color(source_bias_val)
    author_html = f'<span class="card-author">By {row["author"]}</span>' if row.get("author") else ""

    with st.container():
        st.markdown(f"""
        <div class="article-card">
            <div class="card-title">{row['title']}</div>
            <div class="card-meta">
                <span class="card-source">{row['source']}</span> &middot;
                {row['publish_date'].strftime('%B %d, %Y %H:%M')} &middot;
                <span class="card-topic">{row['cluster_label']}</span>
            </div>
            {author_html}
            <div class="bias-bar-wrap">
                <div class="bias-gradient">
                    <div class="bias-marker" style="left: {50 + source_bias_val * 25}%; background: {marker_color};"></div>
                </div>
                <div class="bias-label">Source Bias</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-btn-wrap">', unsafe_allow_html=True)
        if st.button("Read more →", key=f"main_{row['id']}", use_container_width=True):
            st.session_state["article_id"] = int(row['id'])
            st.session_state["return_to"] = "main"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


if not display_df.empty:
    for _, row in display_df.iterrows():
        display_article(row)
else:
    st.info("No articles match the current filters.")
