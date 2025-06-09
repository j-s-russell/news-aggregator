import pandas as pd
import time
import google.generativeai as genai
from config import GOOGLE_API_KEY
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt_tab')


def summarize_abs(text, model):
    prompt = (
        "Please summarize the following news article in 4-5 concise sentences.\n\n"
        f"Article:\n{text}\n\n"
        "Summary:"
    )
    response = model.generate_content(prompt)
    time.sleep(5)
    return response.text.strip()


def summarize_ext(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


def summarize_articles(df):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    df['abs_summary'] = df['content'].apply(lambda x: summarize_abs(x, model))
    df['ext_summary'] = df['content'].apply(lambda x: summarize_ext(x, sentence_count=5))
    
    return df


if __name__ == "__main__":
    from database.db_client import get_unprocessed_articles
    
    df = get_unprocessed_articles()
    df = summarize_articles(df)