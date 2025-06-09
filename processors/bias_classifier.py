import pandas as pd
from transformers import pipeline


def classify_article(text, classifier, pol_labels=["political", "non-political"], bias_labels=["left-wing", "center", "right-wing"]):
    politics_result = classifier(text, candidate_labels=pol_labels)
    is_political = politics_result['labels'][0] == "political"

    if is_political:
        bias_result = classifier(text, candidate_labels=bias_labels)
        bias_scores = bias_result['scores']
    else:
        bias_scores = None

    return bias_scores


def classify_bias(df):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    df['bias_text'] = df['title'].fillna('') + ': ' + df['ext_summary'].fillna('')
    df['article_bias'] = df['bias_text'].apply(lambda x: classify_article(x, classifier))
    
    return df


if __name__ == "__main__":
    from database.db_client import get_unprocessed_articles
    
    df = get_unprocessed_articles()
    df = classify_bias(df)