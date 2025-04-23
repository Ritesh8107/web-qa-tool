import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
# Scraping text Function to extract text from URLs.
def scrape_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error scraping {url}: {e}")
# Function to split text into chunks
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks
# Function to embed chunks
def embed_chunks(chunks):
    embeddings = model.encode(chunks)
    return np.array(embeddings)
# Function to ingest URLs
def ingest_urls(urls):
    all_chunks = []
    for url in urls:
        content = scrape_text(url)
        chunks = chunk_text(content)
        all_chunks.extend(chunks)
    embeddings = embed_chunks(all_chunks)
    return all_chunks, embeddings
# Function to answer questions
def answer_question(question, chunks, embeddings, top_k=3):
    question_embedding = model.encode([question])[0]
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.normalize_L2(np.array([question_embedding]))
    _, indices = index.search(np.array([question_embedding]), top_k)
    relevant = [chunks[i] for i in indices[0]]
    return "\n\n".join(relevant)
