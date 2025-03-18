from agents import Agent, Runner, tool, function_tool
import requests
from bs4 import BeautifulSoup
import faiss
from sentence_transformers import SentenceTransformer
import os
import numpy as np

is_key_set = 'OPENAI_API_KEY' in os.environ

embedding_model = SentenceTransformer("all-MiniLM-L6-v2") 
embedding_size = 384
faiss_index = faiss.IndexFlatL2(embedding_size)
retrieved_texts = [] 


def fetch_webpage_content(url: str) -> str:
    """Fetches and extracts clean text from a webpage."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = "\n".join([p.get_text() for p in soup.find_all("p")])
        return text_content
    else:
        return f"Failed to fetch webpage: {response.status_code}"

def add_to_vector_db(text: str):
    global retrieved_texts
    chunks = text.split(". ")
    chunk_embeddings = embedding_model.encode(chunks)
    faiss_index.add(np.array(chunk_embeddings, dtype=np.float32))
    retrieved_texts.extend(chunks)

def retrieve_relevant_text(query: str, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
    relevant_chunks = [retrieved_texts[idx] for idx in indices[0]]
    return "\n".join(relevant_chunks)

def main():

    #gpt 3.5 used here to reduce tokens
    agent = Agent(name="Assistant", model="gpt-3.5-turbo", instructions="Use the provided context to answer the query.")
    url = "https://en.wikipedia.org/wiki/Henry_VIII"
    text = fetch_webpage_content(url)
    add_to_vector_db(text)
    query = "Was Henry VIII a good person?"
    context = retrieve_relevant_text(query)
    result = Runner.run_sync(agent, input="Provide a concise summary of the kind of leader Henry VII was.", context=context)
    print(result.final_output)


if __name__ == '__main__' and is_key_set:
    main()
else:
    print("Set OPENAI_API_KEY")

