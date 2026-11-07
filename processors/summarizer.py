import pandas as pd
from processors.llm import generate, LLMQuotaError
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
nltk.download('punkt_tab')


def summarize_abs(text):
    prompt = (
        "Please summarize the following news article in 4-5 concise sentences.\n\n"
        f"Article:\n{text[:8000]}\n\n"
        "Summary:"
    )
    summary = generate(prompt, delay=5)
    if summary is None:
        print("  Error summarizing article")
    return summary


def summarize_ext(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


def summarize_articles(df):
    df['ext_summary'] = df['content'].apply(lambda x: summarize_ext(x, sentence_count=5))

    # Abstractive summaries stay NULL on failure so backfill_abs_summaries() retries them
    abs_summaries = []
    for content in df['content']:
        try:
            abs_summaries.append(summarize_abs(content))
        except LLMQuotaError as e:
            print(f"  Gemini quota exhausted, skipping remaining abstractive summaries: {e}")
            abs_summaries += [None] * (len(df) - len(abs_summaries))
            break
    df['abs_summary'] = abs_summaries

    return df


def backfill_abs_summaries(limit=20):
    from database.db_client import get_articles_missing_abs_summary, update_abs_summary

    df = get_articles_missing_abs_summary().head(limit)
    print(f"  {len(df)} articles missing an abstractive summary")
    filled = 0
    for _, row in df.iterrows():
        try:
            summary = summarize_abs(row['content'] or "")
        except LLMQuotaError as e:
            print(f"  Gemini quota exhausted, stopping backfill: {e}")
            break
        if summary:
            update_abs_summary(row['id'], summary)
            filled += 1
    print(f"  Backfilled {filled} abstractive summaries")


if __name__ == "__main__":
    from database.db_client import get_unprocessed_articles
    
    df = get_unprocessed_articles()
    df = summarize_articles(df)
