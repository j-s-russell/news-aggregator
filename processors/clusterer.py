import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from processors.llm import generate, LLMQuotaError
from database.db_client import get_all_articles, replace_articles, insert_cluster_summary, delete_all_cluster_summaries
from processors.chunker import process_article_chunks
from processors.filters import is_sports
import hdbscan
import re
import ast



def clean_labels(output_str):
    """Safely extract dictionary from Gemini's response"""
    
    # Remove markdown code blocks
    cleaned = re.sub(r'```python\s*', '', output_str)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    
    try:
        # Use ast.literal_eval for safety (won't execute arbitrary code)
        mapping_dict = ast.literal_eval(cleaned)
        return mapping_dict
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        print(f"Cleaned text: {cleaned}")
        return None
    
    
def label_cluster(texts, existing_labels=None):
    labels_note = ""
    if existing_labels:
        labels_note = (
            "\nThese topic labels already exist for other clusters: "
            f"{', '.join(existing_labels)}.\n"
        )
    prompt = (
        "You are given a list of news article descriptions. "
        "Please respond with a **single word or short phrase** that summarizes the main topic of the cluster."
        "Do NOT include an 'Other', 'News', 'Headlines' or very general category names. "
        "Avoid generic umbrella terms like 'Sports', 'Politics', 'Business' — prefer specific "
        "subtopics like 'Immigration Policy', 'Presidential Election', 'Tech Stocks'."
        f"{labels_note}"
        "Choose a label that is specific and does NOT duplicate or overlap with existing labels.\n\n"
        f"Articles:\n{texts}\n\n"
        "Topic label:"
    )
    label = generate(prompt)
    if label is None:
        print("  Error labeling cluster")
    return label

def normalize_labels(unique_labels):
    prompt = f"""
    I have these news article cluster labels: {unique_labels}

    Consolidate labels that describe the SAME topic into one canonical label.
    Example: "Athletics" and "Sports" both describe general sports -> map both to "Sports".
    Example: "Legal/Political" and "Politics" both describe political news -> map both to "Politics".
    Do NOT merge labels describing genuinely different subtopics (e.g., "NFL Football" and
    "College Athletics" stay separate; "Ukraine War" and "Middle East Conflict" stay separate).

    Map every label to its canonical category. Provide only a Python dictionary, no other text.

    Format: {{"original_label": "canonical_category", ...}}
    """
    for attempt in range(2):
        response = generate(prompt)
        label_map = clean_labels(response) if response else None
        if label_map and set(label_map) >= set(unique_labels):
            return label_map
        print(f"  Error normalizing labels (attempt {attempt+1})")
    return None


def cluster_articles(method='kmeans', normalize=False, reduce_dim=False):
    df = get_all_articles()

    # Drop sports rows; replace_articles() below rewrites the table, purging them from the DB
    sports_mask = df.apply(is_sports, axis=1)
    print(f"Purging {sports_mask.sum()} sports articles from DB")
    df = df[~sports_mask].reset_index(drop=True)

    # Define clusters
    df['cluster_text'] = df['title'].fillna('') + '. ' + df['ext_summary'].fillna('')
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = st_model.encode(df['cluster_text'].tolist(), normalize_embeddings=True)

    if reduce_dim:
        pca = PCA(n_components=10, random_state=42)
        embeddings = pca.fit_transform(embeddings)
        # Normalize after PCA to maintain unit norm for cosine distance
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings)
    df['embedding'] = embeddings.tolist()

    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=0.2
        )
        
        df['cluster'] = clusterer.fit_predict(embeddings)
    
    else:
        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(embeddings)
    
    # Label clusters (largest first so the biggest cluster claims the canonical term)
    cluster_labels = {}
    unique_labels = set()
    cluster_rank = (
        df['cluster'].value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )
    for cluster_id in cluster_rank:
        cluster_summaries = df[df['cluster'] == cluster_id]['cluster_text'].tolist()
        sampled = cluster_summaries[:15]
        text_block = "\n".join(s[:300] for s in sampled)
        label = label_cluster(text_block, existing_labels=list(unique_labels))
        if label is None:
            # Don't overwrite the existing labels with placeholders
            raise RuntimeError(f"Labeling failed for cluster {cluster_id}; leaving existing labels untouched")
        if label in unique_labels:
            label = f"{label}-{cluster_id}"
        unique_labels.add(label)
        cluster_labels[cluster_id] = label
        
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    if normalize:
        try:
            label_map = normalize_labels(unique_labels)
            if label_map:
                df['cluster_label'] = df['cluster_label'].map(label_map).fillna(df['cluster_label'])
        except Exception as e:
            print(f"Error: {e}. Returning unnormalized labels.")

    print(df['cluster_label'].unique())
    print(df['cluster_label'].value_counts())
    
    replace_articles(df)

    print("Generating cluster summaries...")
    summaries = {}
    for label in df['cluster_label'].unique():
        try:
            summary = generate_cluster_summary(label, df)
        except LLMQuotaError as e:
            print(f"  Gemini quota exhausted, stopping cluster summaries: {e}")
            break
        if summary:
            summaries[label] = summary

    if summaries:
        print(f"Replacing cluster summaries ({len(summaries)} generated)...")
        delete_all_cluster_summaries()
        for label, summary in summaries.items():
            insert_cluster_summary(label, summary)
    else:
        print("No cluster summaries generated; keeping existing ones.")
    print("Done.")

    print("Chunking articles...")
    for _, row in df.iterrows():
        process_article_chunks(row['id'], row['ext_summary'])
    print("Done.")


def generate_cluster_summary(cluster_label, articles_df):
    cluster_arts = articles_df[articles_df['cluster_label'] == cluster_label]
    if cluster_arts.empty:
        return None

    articles_text = ""
    sampled = cluster_arts.sample(n=min(8, len(cluster_arts)), random_state=42)
    for _, row in sampled.iterrows():
        summary = str(row.get("ext_summary", ""))[:200]
        articles_text += f"- {row['title']} ({row['source']}): {summary}\n"

    prompt = (
        f"You are summarizing a news topic labeled \"{cluster_label}\".\n"
        "Below are articles that belong to this topic.\n\n"
        f"Articles:\n{articles_text}\n\n"
        f"Write a 4-5 sentence overview focused ONLY on the \"{cluster_label}\" topic. "
        "Do not include information about other subjects or tangentially related stories. "
        "Be factual and concise. Do not use bullet points."
    )

    summary_text = generate(prompt)
    if summary_text is None:
        print(f"  Error generating summary for '{cluster_label}'")
    return summary_text


if __name__ == "__main__":
    cluster_articles(normalize=True)
    
    
    
    