import streamlit as st
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# ---------- CONFIGURATION ----------
load_dotenv()
                                      
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DBNAME")

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
    query = "SELECT * FROM news_pipeline ORDER BY publish_date DESC LIMIT 200;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="🗞️ News Aggregator", layout="wide")
st.title("🗞️ News Aggregator")
st.write("Browse articles stored in your Supabase PostgreSQL database.")

# Load articles
articles = get_articles()

# Optional filters
sources = st.multiselect("Filter by source", options=articles['source'].unique(), default=articles['source'].unique())
keywords = st.text_input("Search in title or description")

# Filter logic
filtered = articles[articles['source'].isin(sources)]

if keywords:
    keyword_lower = keywords.lower()
    filtered = filtered[
        filtered['title'].str.lower().str.contains(keyword_lower) |
        filtered['description'].str.lower().str.contains(keyword_lower)
    ]

# Display articles
for _, row in filtered.iterrows():
    st.markdown(f"### [{row['title']}]({row['url']})")
    st.write(f"*{row['source']}* • {row['publish_date'].strftime('%B %d, %Y %H:%M')}")
    st.write(row['description'])
    
    # Bias bar
    bias_val = row['bias']
    marker_color = get_bias_color(bias_val)

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
                left: {50 + bias_val * 25}%;
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
    
    if row.get("author"):
        st.caption(f"By {row['author']}")
    st.markdown("---")