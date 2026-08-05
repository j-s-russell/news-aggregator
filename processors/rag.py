from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import DEEPSEEK_API_KEY
from database.db_client import retrieve_chunks

st_model = SentenceTransformer('all-MiniLM-L6-v2')

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = (
    "You are a helpful news assistant. Answer questions using only the "
    "provided context from news articles. If the context doesn't contain "
    "enough information to answer the question, say so clearly. Be concise "
    "and factual. When referencing a specific article, mention its source."
    "While your answers should be based on the provided context, do not "
    "use the phrase 'based on the provided context' or anything similar."
)


def build_context(chunks):
    lines = []
    for chunk in chunks:
        title = chunk.get("title", "Unknown")
        source = chunk.get("source", "Unknown")
        text = chunk.get("chunk_text", "")
        lines.append(f"[Article: {title} ({source})]\n{text}")
    return "\n---\n".join(lines)


def answer_question(question, chat_history=None, match_count=5):
    query_embedding = st_model.encode([question])[0].tolist()
    chunks = retrieve_chunks(query_embedding, match_count=match_count)

    context = build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content
