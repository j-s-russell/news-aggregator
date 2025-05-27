import pandas as pd
import requests
import time
import psycopg2
from dotenv import load_dotenv
import os
from transformers import pipeline
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


api_key = 'be3e1169193a4f7abafb84d3d1d0ef10'
url = 'https://newsapi.org/v2/everything'

# Settings
category = 'politics'
page_size = 5
max_articles = 10

left_sources = ['cnn', 'msnbc', 'the-huffington-post']
right_sources = ['fox-news', 'the-washington-times', 'breitbart-news', 'national-review']
center_sources = ['reuters', 'bbc-news', 'associated-press', 'bloomberg', 'usa-today', 'the-hill']
other_sources = ['abc-news', 'cbs-news', 'axios']

sources = left_sources + right_sources + center_sources + other_sources

bias_dict = {'cnn': -1,
             'msnbc': -2,
             'the-huffington-post': -2,
             'fox-news': 2,
             'the-washington-times': 1,
             'breitbart-news': 2,
             'reuters': 0,
             'bbc-news': 0,
             'associated-press': -2,
             'bloomberg': -1,
             'usa-today': -1,
             'the-hill': 0,
             'abc-news': -1,
             'axios': -1,
             'cbs-news': -1,
             'national-review': 1}

def get_bias_value(source):
    return bias_dict.get(source, 0)


# Gather articles
articles = []

for source in sources:    
    for page in range(1, (max_articles // page_size) + 1):
        params = {
            'apiKey': api_key,
            'sources': source,
            'pageSize': page_size,
            'page': page,
            'language': 'en',
        }
    
    
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code != 200 or 'articles' not in data:
            print(f"Error on page {page}: {data.get('message', 'Unknown error')}")
            break
    
        if not data['articles']:
            break
        
        articles.extend(data['articles'])
        
        time.sleep(1)
    
    time.sleep(1)
    
df = pd.DataFrame(articles)
df['source'] = df['source'].apply(lambda x: x['id'])
df['bias'] = df['source'].apply(get_bias_value)


# Similar articles (similarity threshold TBD)


# Connect to database
             
load_dotenv()
                                      
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

connection = psycopg2.connect(
    user=USER,
    password=PASSWORD,
    host=HOST,
    port=PORT,
    dbname=DBNAME
)

cursor = connection.cursor()

# Insert articles
    
for _, row in df.iterrows():
    title = row['title']
    author = row['author']
    source = row['source']
    description = row['description']
    url = row['url']
    publish_date = row['publishedAt']
    content = row['content']
    bias = row['bias']

    cursor.execute("""
        INSERT INTO news_pipeline (title, author, source, description, url, publish_date, content, bias)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING
    """, (
        title,
        author,
        source,
        description,
        url,
        publish_date,
        content,
        bias
    ))
    
    connection.commit()
    

# Delete extra articles

cursor.execute("""
    DELETE FROM news_pipeline
    WHERE id IN (
        SELECT id FROM news_pipeline
        ORDER BY publish_date DESC
        OFFSET 100
    )
""")
connection.commit()

cursor.close()
connection.close()
    
    
    
    
    