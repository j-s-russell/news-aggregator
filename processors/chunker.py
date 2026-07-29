from sentence_transformers import SentenceTransformer
from database.db_client import insert_chunks

st_model = SentenceTransformer('all-MiniLM-L6-v2')


def chunk_text(content, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        chunks.append(chunk)
        start = end - overlap
    return [(chunk, i) for i, chunk in enumerate(chunks)]


def process_article_chunks(article_id, content):
    if not content or len(content.strip()) == 0:
        return

    chunks = chunk_text(content)
    texts = [c[0] for c in chunks]
    embeddings = st_model.encode(texts)

    enriched = [
        (text, idx, emb.tolist())
        for (text, idx), emb in zip(chunks, embeddings)
    ]
    insert_chunks(article_id, enriched)
