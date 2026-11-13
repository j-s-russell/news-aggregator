import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from newspaper import Article
import psycopg2
from config import *
from database.db_client import insert_raw_articles
from processors.filters import is_sports


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
    page_size = 15
    max_articles = 15
    
    # AP News and The Hill are excluded: they block scrapers and opt out of AI use in robots.txt
    left_sources = ['cnn', 'msnbc', 'the-huffington-post', 'politico', 'the-washington-post', 'time']
    right_sources = ['fox-news', 'the-washington-times', 'breitbart-news', 'national-review', 'the-american-conservative']
    center_sources = ['bbc-news', 'usa-today', 'the-wall-street-journal', 'newsweek']
    other_sources = ['abc-news', 'cbs-news', 'axios', 'nbc-news', 'al-jazeera-english']
    sources = left_sources + right_sources + center_sources + other_sources

    # Bias scores from www.allsides.com
    bias_dict = {'cnn': -1,
                 'msnbc': -2,
                 'the-huffington-post': -2,
                 'politico': -1,
                 'the-washington-post': -1,
                 'time': -1,
                 'fox-news': 2,
                 'the-washington-times': 1,
                 'breitbart-news': 2,
                 'national-review': 1,
                 'the-american-conservative': 1,
                 'bbc-news': 0,
                 'the-wall-street-journal': 0,
                 'usa-today': -1,
                 'newsweek': 0,
                 'abc-news': -1,
                 'axios': -1,
                 'cbs-news': -1,
                 'nbc-news': -1,
                 'al-jazeera-english': -1}
    
    def get_bias_value(source):
        return bias_dict.get(source, 0)
    
    
    
    # Top Headlines
    
    top_url = 'https://newsapi.org/v2/top-headlines'
    page_size = 15
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
    
    
        response = requests.get(top_url, params=params)
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
    
    
    
    
    # All other articles
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

    # Drop sports before scraping so they never reach Gemini
    sports_mask = df.apply(is_sports, axis=1)
    print(f"Dropping {sports_mask.sum()} sports articles")
    df = df[~sports_mask].copy()

    df['content'] = df['url'].apply(scrape)
    df = df.sort_values(by='top', ascending=False)
    df = df.drop_duplicates(subset='url', keep='first')
    df = df.drop_duplicates(subset='title', keep='first')
    df = df[df['content'].str.len().between(1000, 12000)]
    df = df[~df['content'].str.contains(r'\blive\b', case=False, na=False, regex=True)]
    df = df[~df['title'].str.contains('video', case=False, na=False)]
    df = df.reset_index(drop=True)
    
    insert_raw_articles(df)



