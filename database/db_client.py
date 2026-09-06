import psycopg2
import psycopg2.extras
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import USER, PASSWORD, HOST, PORT, DBNAME


# def insert_raw_articles(df):
#     connection = psycopg2.connect(
#         user=USER,
#         password=PASSWORD,
#         host=HOST,
#         port=PORT,
#         dbname=DBNAME
#     )
    
#     cursor = connection.cursor()
        
#     for _, row in df.iterrows():
#         title = row['title']
#         author = row['author']
#         source = row['source']
#         description = row['description']
#         url = row['url']
#         publish_date = row['publish_date']
#         content = row['content']
#         source_bias = row['source_bias']
#         top = row['top']
    
#         cursor.execute("""
#             INSERT INTO news_pipeline (title, author, source, description, url, publish_date, content, source_bias, top)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#             ON CONFLICT (url) DO NOTHING
#         """, (
#             title,
#             author,
#             source,
#             description,
#             url,
#             publish_date,
#             content,
#             source_bias,
#             top
#         ))
        
#         connection.commit()
    
#     cursor.close()
#     connection.close()

def insert_raw_articles(df):
    df = df.where(pd.notnull(df), None)  # convert NaN -> NULL for Postgres

    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    try:
        cursor = connection.cursor()

        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO news_pipeline (title, author, source, description, url, publish_date, content, source_bias, top)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                """, (
                    row['title'],
                    row['author'],
                    row['source'],
                    row['description'],
                    row['url'],
                    row['publish_date'],
                    row['content'],
                    row['source_bias'],
                    row['top']
                ))
                connection.commit()
            except Exception as e:
                connection.rollback()
                print(f"Failed to insert article with url={row.get('url')}: {e}")

        cursor.close()
    finally:
        connection.close()
    

def get_all_articles():
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
    """
    
    df = pd.read_sql(query, connection)
    connection.close()
    return df


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
        ext_summary = row['ext_summary']
    
        cursor.execute("""
            INSERT INTO news_pipeline (title, author, source, description, url, publish_date, content, source_bias, top, abs_summary, cluster_label, embedding, ext_summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            ext_summary
        ))
        
        connection.commit()
        
    
    # Delete extra articles
    

    cursor.execute("""
        DELETE FROM news_pipeline
        WHERE publish_date < NOW() - INTERVAL '4 days'
    """)
    connection.commit()
    
    cursor.close()
    connection.close()
    
    

def replace_articles(df):
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    
    cursor = connection.cursor()
    
    try:
        # Delete all existing articles first
        cursor.execute("DELETE FROM news_pipeline")
        print(f"Deleted all existing articles from database")
        
        # Insert all articles from the dataframe
        for _, row in df.iterrows():
            row_id = row['id']
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
            ext_summary = row['ext_summary']

            cursor.execute("""
                INSERT INTO news_pipeline (id, title, author, source, description, url, publish_date, content, source_bias, top, abs_summary, cluster_label, embedding, ext_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row_id,
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
                ext_summary
            ))
        
        # Commit all changes at once
        connection.commit()
        print(f"Inserted {len(df)} articles into database")
        
    except Exception as e:
        # Rollback in case of error
        connection.rollback()
        print(f"Error inserting articles: {e}")
        raise
    finally:
        cursor.close()
        connection.close()


def insert_cluster_summary(cluster_label, summary_text):
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO cluster_summaries (cluster_label, summary_text)
        VALUES (%s, %s)
        ON CONFLICT (cluster_label) DO UPDATE SET
            summary_text = EXCLUDED.summary_text,
            generated_at = now()
    """, (cluster_label, summary_text))

    connection.commit()
    cursor.close()
    connection.close()


def delete_all_cluster_summaries():
    connection = psycopg2.connect(
        user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME
    )
    cursor = connection.cursor()
    cursor.execute("DELETE FROM cluster_summaries")
    connection.commit()
    cursor.close()
    connection.close()


def get_cluster_summary(cluster_label):
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
        SELECT * FROM cluster_summaries WHERE cluster_label = %s
    """, (cluster_label,))

    row = cursor.fetchone()
    cursor.close()
    connection.close()

    if row is None:
        return None

    return dict(row)


def insert_chunks(article_id, chunks):
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    cursor = connection.cursor()
    args_str = ",".join(
        cursor.mogrify(
            "(%s, %s, %s, %s::vector)",
            (article_id, text, idx, emb)
        ).decode() for text, idx, emb in chunks
    )
    cursor.execute(f"""
        INSERT INTO article_chunks (article_id, chunk_text, chunk_index, embedding)
        VALUES {args_str}
    """)

    connection.commit()
    cursor.close()
    connection.close()


def get_chunks_by_article(article_id):
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
        SELECT * FROM article_chunks WHERE article_id = %s ORDER BY chunk_index
    """, (article_id,))

    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    return [dict(row) for row in rows]


def retrieve_chunks(query_embedding, match_count=5):
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
        SELECT * FROM match_chunks(%s::vector, %s)
    """, (str(query_embedding), match_count))

    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    return [dict(row) for row in rows]