import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = 'multi-qa-MiniLM-L6-cos-v1' # You can change to other models
model = SentenceTransformer(MODEL_NAME)

def is_valid_url(url):
    """Checks if the url is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def fetch_and_extract_text(url):
    """Fetches the url and extracts text"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(
            p.get_text(strip=True)
            for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'td'])
        )
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
def find_best_match(query, cdp_base_url):
    print(f"Fetching data from: {cdp_base_url}")
    text_content = fetch_and_extract_text(cdp_base_url)
    if not text_content:
        print("No content fetched from the URL")
        return None
    
    print(f"Fetched text: {text_content[:500]}...")  # Show first 500 characters


    query_embedding = model.encode(preprocess_text(query))
    sentences = text_content.split('.')
    sentence_similarities = []
    for sentence in sentences:
        if sentence := sentence.strip():
            sentence_embedding = model.encode(preprocess_text(sentence))
            similarity = cosine_similarity([query_embedding], [sentence_embedding])[0][0]
            sentence_similarities.append((sentence, similarity))
            
    
    top_sentences = sorted(sentence_similarities, key=lambda x: x[1], reverse=True)[:3]
    best_match = ".\n".join([sentence for sentence, score in top_sentences])

    print(f"Best Match Found: {best_match}")

    return best_match

def clear_data_directory(base_dir='data'):
    """Clears the data directory, deleting all files within subdirectories"""
    for cdp in os.listdir(base_dir):
        cdp_path = os.path.join(base_dir, cdp)
        if os.path.isdir(cdp_path):
            for filename in os.listdir(cdp_path):
                file_path = os.path.join(cdp_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

