import argparse
from retrievers.fetcher import fetch_articles
from processors.summarizer import summarize_articles
from processors.clusterer import cluster_articles
from processors.bias_classifier import classify_bias
from database.db_client import get_unprocessed_articles, remove_unprocessed_articles, insert_articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-only', action='store_true')
    parser.add_argument('--process-only', action='store_true')
    args = parser.parse_args()
    
    if args.fetch_only or not args.process_only:
        print("Fetching articles...")
        fetch_articles()
    
    if args.process_only or not args.fetch_only:
        print("Processing articles...")
        df = get_unprocessed_articles()
        print("Summarizing articles...")
        df = summarize_articles(df)
        print("Classifying articles...")
        df = classify_bias(df)
        print("Clustering articles...")
        df = cluster_articles(df)
        print("Removing unprocessed articles...")
        remove_unprocessed_articles()
        print("Inserting articles...")
        insert_articles(df)

if __name__ == "__main__":
    main()