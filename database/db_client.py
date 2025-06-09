import psycopg2
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import USER, PASSWORD, HOST, PORT, DBNAME


def get_unprocessed_articles():
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    
    query = """
    SELECT *
    FROM news_pipeline
    WHERE abs_summary IS NULL
      AND ext_summary IS NULL
      AND article_bias IS NULL
      AND cluster_label IS NULL
    ORDER BY publish_date DESC
    LIMIT 500;
    """
    
    df = pd.read_sql(query, connection)
    connection.close()
    return df

def remove_unprocessed_articles():
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    
    cursor = connection.cursor()
    cursor.execute("""
        DELETE FROM news_pipeline
        WHERE abs_summary IS NULL
          AND ext_summary IS NULL
          AND article_bias IS NULL
          AND cluster_label IS NULL
    """)
    connection.commit()
    
    cursor.close()
    connection.close()


def insert_articles(df):
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    
    cursor = connection.cursor()
        
    for _, row in df.iterrows():
        title = row['title']
        author = row['author']
        source = row['source']
        description = row['description']
        url = row['url']
        publish_date = row['publish_date']
        content = row['content']
        source_bias = row['source_bias']
        top = row['top']
        abs_summary = row['abs_summary']
        cluster_label = row['cluster_label']
        embedding = row['embedding']
        article_bias = row['article_bias']
        ext_summary = row['ext_summary']
    
        cursor.execute("""
            INSERT INTO news_pipeline (title, author, source, description, url, publish_date, content, source_bias, top, abs_summary, cluster_label, embedding, article_bias, ext_summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING
        """, (
            title,
            author,
            source,
            description,
            url,
            publish_date,
            content,
            source_bias,
            top,
            abs_summary,
            cluster_label,
            embedding,
            article_bias,
            ext_summary
        ))
        
        connection.commit()
        
    
    # Delete extra articles
    
    cursor.execute("""
        DELETE FROM news_pipeline
        WHERE id IN (
            SELECT id FROM news_pipeline
            ORDER BY publish_date DESC
            OFFSET 400
        )
    """)
    connection.commit()
    
    cursor.close()
    connection.close()
    
    
    
    