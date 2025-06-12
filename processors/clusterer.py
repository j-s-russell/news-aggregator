import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import GOOGLE_API_KEY
import google.generativeai as genai
import time
from database.db_client import get_all_articles, replace_articles
import hdbscan
import re
import ast



genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


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
    
    
def label_cluster(texts):
    prompt = (
        "You are given a list of news article descriptions. "
        "Please respond with a **single word or short phrase** that summarizes the main topic of the cluster."
        "Do NOT include an 'Other', 'News', or very general category. \n\n"
        f"Articles:\n{texts}\n\n"
        "Topic label:"
    )
    response = model.generate_content(prompt)
    time.sleep(4.1)
    return response.text.strip()

def normalize_labels(unique_labels):
    prompt = f"""
    I have these news article cluster labels: {unique_labels}
    
    I want to merge duplicate into single categories.
    
    These labels need to describe the same topic, not just similar topics ("Legal/Political" and "Politics", "Basketball" and "NBA", etc.)
    
    Please provide a Python dictionary mapping each label to the most appropriate main category.
    Only return the dictionary, no other text.
    
    Format: {{"original_label": "main_category", ...}}
    """
    response = model.generate_content(prompt)
    time.sleep(4.1)
    label_map = clean_labels(response.text)
    return label_map



def cluster_articles(method='kmeans', normalize=False):
    df = get_all_articles()
    
    # Define clusters
    df['cluster_text'] = df['title'].fillna('') + '. ' + df['ext_summary'].fillna('')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['cluster_text'])

    pca = PCA(n_components=10, random_state=42)
    embeddings = pca.fit_transform(embeddings)
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
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
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(embeddings)
    
    # Label clusters
    cluster_labels = {}
    unique_labels = set()
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_summaries = df[df['cluster'] == cluster_id]['cluster_text'].tolist()
        text_block = "\n".join(cluster_summaries)
        label = label_cluster(text_block)
        unique_labels.add(label)
        cluster_labels[cluster_id] = label
        
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    if normalize:
        try:
            label_map = normalize_labels(unique_labels)
            df['cluster_label'] = df['cluster_label'].map(label_map).fillna(df['cluster_label'])
        except Exception as e:
            print(f"Error: {e}. Returning unnormalized labels.")
    
    # Add default categories
    categories = df['cluster_label'].unique().tolist()
    default_categories = ['Pop Culture', 'Science']
    categories = categories + default_categories
    
    # Reassign to new labels
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    def assign_category(text):
        result = classifier(text, candidate_labels=categories)
        return result['labels'][0]
    
    df['cluster_label'] = df['cluster_text'].apply(assign_category)
    
    # Map labels to dataframe
    print(df['cluster_label'].unique())
    print(df['cluster_label'].value_counts())
    
    replace_articles(df)
    
    
if __name__ == "__main__":
    cluster_articles()
    
    
    
    