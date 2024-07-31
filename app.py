import gradio as gr
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import requests
from dotenv import find_dotenv, load_dotenv
import textwrap
import os
import json

# Load environment variables from the .env file
# load_dotenv("key.env")
# gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_api_key = "Put your api key here!"

# Initialize the Hugging Face model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_db_from_youtube_video_url(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # Generate embeddings for the documents
    docs_content = [doc.page_content for doc in docs]
    embeddings = embedding_model.encode(docs_content)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return docs, index

def get_response_from_query(docs, index, query, k=4):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    docs_page_content = " ".join([docs[idx].page_content for idx in indices[0]])

    # Use Gemini API for generating response
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {"parts": [{"text": f"Question: {query}\nDocs: {docs_page_content}"}]}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}", None
    except ValueError as e:
        return f"JSON decoding error: {e} - Response content: {response.text}", None

    print("Full Response Data:", json.dumps(response_data, indent=2))  # Log the entire response

    if response.status_code != 200:
        return f"Error: {response_data.get('error', {}).get('message', 'Unknown error')}", None

    # Access the generated text correctly
    try:
        generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError) as e:
        return f"Error accessing response content: {e}", None

    return generated_text, docs

def process_inputs(name, link):
    try:
        video_url = link
        query = name

        docs, index = create_db_from_youtube_video_url(video_url)
        response, docs = get_response_from_query(docs, index, query)
        if docs is None:  # Handle errors from get_response_from_query
            return response
        return textwrap.fill(response, width=85)
    except Exception as e:
        return str(e)

inputs = [
    gr.Textbox(label="Query"),
    gr.Textbox(label="Link")
]

outputs = gr.Textbox(label="Output")

demo = gr.Interface(fn=process_inputs, inputs=inputs, outputs=outputs)
demo.launch()
