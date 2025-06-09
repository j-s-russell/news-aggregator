import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from config import GOOGLE_API_KEY
import google.generativeai as genai
import time


genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def label_cluster(texts):
    prompt = (
        "You are given a list of news article descriptions. "
        "Please respond with a **single word or short phrase** that summarizes the main topic of the cluster. If there's various topics, label it 'Other'.\n\n"
        f"Articles:\n{texts}\n\n"
        "Topic label:"
    )
    response = model.generate_content(prompt)
    time.sleep(4)
    return response.text.strip()



def cluster_articles(df, method='kmeans'):
    
    # Define clusters
    df['cluster_text'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['cluster_text'])

    pca = PCA(n_components=10, random_state=42)
    embeddings = pca.fit_transform(embeddings)
    df['embedding'] = embeddings.tolist()

    if method == 'kmeans':
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(embeddings)
    
    else:
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        df['cluster'] = dbscan.fit_predict(embeddings)
        
    # Label clusters
    cluster_labels = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_summaries = df[df['cluster'] == cluster_id]['cluster_text'].tolist()
        text_block = "\n".join(cluster_summaries)
        label = label_cluster(text_block)
        cluster_labels[cluster_id] = label
    
    # Map labels to dataframe
    df['cluster_label'] = df['cluster'].map(cluster_labels)
    
    return df
    
    
    
    