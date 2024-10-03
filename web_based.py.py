import requests
from flask import Flask, request, jsonify, render_template
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import numpy as np
from ast import literal_eval

app = Flask(__name__)

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Gemini API key (replace with your actual key)
gemini_api_key = 'AIzaSyB5LRsvv_8MuHB8fw843B7O1AZ69cn06eo'

# Set up headers for Gemini API requests
headers = {
    'Authorization': f'Bearer {gemini_api_key}',
    'Content-Type': 'application/json'
}

class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

# ... (other functions remain unchanged)

def get_hyperlinks(url):
    try:
        with urllib.request.urlopen(url) as response:
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []
    
    parser = HyperlinkParser()
    parser.feed(html)
    return parser.hyperlinks

def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link
        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)
    return list(set(clean_links))

def crawl(url):
    local_domain = urlparse(url).netloc
    queue = deque([url])
    seen = set([url])

    if not os.path.exists("text/"):
        os.mkdir("text/")
    if not os.path.exists(f"text/{local_domain}/"):
        os.mkdir(f"text/{local_domain}/")
    if not os.path.exists("processed"):
        os.mkdir("processed")

    while queue:
        url = queue.pop()
        print(f"Crawling: {url}")  # Debug message
        with open(f'text/{local_domain}/{url[8:].replace("/", "_")}.txt', "w", encoding='utf-8') as f:
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            text = soup.get_text()
            if "You need to enable JavaScript to run this app." in text:
                print(f"Unable to parse page {url} due to JavaScript being required")
            f.write(text)

        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ', regex=True)
    serie = serie.str.replace('\\n', ' ', regex=True)
    serie = serie.str.replace('  ', ' ', regex=True)
    return serie

# Getting embedding for a sample question
def get_embedding(text):
    print(f"Getting embeddings for question: {text}")
    try:
        response = requests.post(
            # "https://api.gemini.com/v1/text/embeddings",  # Replace with Gemini's endpoint
            # "https://api.gemini.com/v1/completions",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyB5LRsvv_8MuHB8fw843B7O1AZ69cn06eo",
            headers=headers,
            json={"inputs": [text], "model": "text-embedding-model"}
        )
        
        # Debug print to inspect the response
        print("Response status code:", response.status_code)
        print("Response body:", response.text)

        if response.status_code == 200:
            response_data = response.json()
            if 'data' in response_data and len(response_data['data']) > 0:
                return response_data['data'][0]['embedding']
            else:
                print("Unexpected response structure:", response_data)
                return None
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def calculate_distance(embedding_a, embedding_b):
    # Check for None values
    if embedding_a is None or embedding_b is None:
        raise ValueError("One of the embeddings is None.")
    
    # Convert to numpy arrays
    a = np.array(embedding_a)
    b = np.array(embedding_b)

    # Compute cosine similarity
    cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - cosine_similarity  # Return distance (1 - similarity)

def distances_from_embeddings(query_embedding, embeddings):
    distances = []
    for embedding in embeddings:
        # Check for None embedding
        if embedding is None:
            distances.append(float('inf'))  # You can choose a suitable value or handling method
            continue
        distance = calculate_distance(query_embedding, embedding)
        distances.append(distance)
    return distances

def create_context(question, df, max_len=1800):
    # Get query embedding
    q_embeddings = get_embedding(question)  
    
    # If the query embedding is None, raise an exception
    if q_embeddings is None:
        raise ValueError("Query embedding is None. Please check the embedding process.")
    
    # Remove rows where embeddings are None
    df = df[df['embeddings'].notna()]
    
    # Calculate distances only for valid embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values)

    returns = []
    cur_len = 0
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])
    
    return "\n\n###\n\n".join(returns)

# def create_context(question, df, max_len=1800):
#     q_embeddings = get_embedding(question)  # Ensure this function is defined and returns valid embeddings
#     if not question or not isinstance(question, str):
#         raise ValueError("Invalid question input.")

#     # if q_embeddings is None:
#     #     raise ValueError("Query embedding is None.")
    
#     df = df[df['embeddings'].notna()]  # Remove rows with NaN embeddings
#     df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values)
#     returns = []
#     cur_len = 0
#     for i, row in df.sort_values('distances', ascending=True).iterrows():
#         cur_len += row['n_tokens'] + 4
#         if cur_len > max_len:
#             break
#         returns.append(row["text"])
#     return "\n\n###\n\n".join(returns)

@app.route('/crawl', methods=['POST'])
def crawl_endpoint():
    data = request.get_json()
    url = data.get('url')
    if url:
        crawl(url)
        return jsonify({"message": "Crawling completed successfully."}), 200
    return jsonify({"error": "URL not provided"}), 400

########################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.form
    question = data.get('question')
    
    # Load the DataFrame
    df = pd.read_csv('processed/embeddings.csv', index_col=0)

    # Check for and display rows with NaN in 'embeddings'
    print("Rows with NaN in 'embeddings':")
    print(df[df['embeddings'].isna()])

    # Replace NaN values with empty list or some default value
    df['embeddings'] = df['embeddings'].fillna('[]')  # Replaces NaN with '[]'

    # Convert the embeddings column from string to list safely
    try:
        df['embeddings'] = df['embeddings'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    except Exception as e:
        return render_template('result.html', question=question, answer=f"Error processing embeddings: {str(e)}")

    if question:
        context = create_context(question, df)
        try:
            # Call your API or processing logic here
            # For example:
            response = requests.post(
                "https://api.gemini.com/v1/completions",
                headers=headers,
                json={
                    "prompt": f"Answer the question based on the context below: {context}",
                    "model": "text-completion-model"
                }
            )
            answer = response.json()['choices'][0]['text']
            return render_template('result.html', question=question, answer=answer)
        except Exception as e:
            return render_template('result.html', question=question, answer=f"Error: {str(e)}")
    return render_template('result.html', question=question, answer="Question not provided")

if __name__ == '__main__':
    app.run(debug=True)