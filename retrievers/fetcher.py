import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from newspaper import Article
import psycopg2
from config import *
from database.db_client import insert_raw_articles


def scrape(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.google.com",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    time.sleep(1)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        article = Article(url)
        article.set_html(response.text)
        article.parse()

        return article.text

    except Exception as e:
        print(f"Error downloading/parsing {url}: {e}")
        return ""
    
    

def fetch_articles():    
    url = 'https://newsapi.org/v2/everything'
    
    # Settings
    category = 'politics'
    page_size = 5
    max_articles = 10
    
    left_sources = ['cnn', 'msnbc', 'the-huffington-post']
    right_sources = ['fox-news', 'the-washington-times', 'breitbart-news', 'national-review']
    center_sources = ['reuters', 'bbc-news', 'associated-press', 'usa-today']
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
    
    
    
    # Top Headlines
    
    top_url = 'https://newsapi.org/v2/top-headlines'
    page_size = 5
    max_top_articles = 20
    top_sources = ",".join(sources)
    
    top_articles = []
    
    for page in range(1, (max_articles // page_size) + 1):
        params = {
            'apiKey': NEWS_API_KEY,
            'sources': top_sources,
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
        
        top_articles.extend(data['articles'])
        
        time.sleep(1)
        
    top_df = pd.DataFrame(top_articles)
    top_df['top'] = True
    
    # Everything
    all_articles = []
    
    for source in sources:    
        for page in range(1, (max_articles // page_size) + 1):
            params = {
                'apiKey': NEWS_API_KEY,
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
            
            all_articles.extend(data['articles'])
            
            time.sleep(1)
        
        time.sleep(1)
        
    all_df = pd.DataFrame(all_articles)
    all_df['top'] = False
    
    df = pd.concat([top_df, all_df], ignore_index=True)
    
    df = df.rename(columns={'publishedAt': 'publish_date'})
    df['source'] = df['source'].apply(lambda x: x['id'])
    df['source_bias'] = df['source'].apply(get_bias_value)
    df['content'] = df['url'].apply(scrape)
    df = df.sort_values(by='top', ascending=False)
    df = df.drop_duplicates(subset='url', keep='first')
    df = df.drop_duplicates(subset='title', keep='first')
    df = df[df['content'].str.len().between(1000, 12000)]
    df = df[~df['content'].str.contains('live', case=False, na=False)]
    df = df[~df['title'].str.contains('video', case=False, na=False)]
    df = df.reset_index(drop=True)
    
    insert_raw_articles(df)



