import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Gemini
from PyPDF2 import PdfReader
from arxiv import Search, SortCriterion
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

# Set up Google API Key directly in the code
GOOGLE_API_KEY = "AIzaSyCSOt-RM3M-SsEQObh5ZBe-XwDK36oD3lM"

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=GOOGLE_API_KEY,  # Pass the API key here
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def download_arxiv_pdf(arxiv_url: str, download_dir: str = "./pdfs/") -> str:
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    search = Search(query=arxiv_url, sort_by=SortCriterion.Relevance)
    results = list(search.results())
    
    if not results:
        return None, f"Failed to find paper with URL: {arxiv_url}"
    
    paper = results[0]
    pdf_url = paper.pdf_url
    response = requests.get(pdf_url)
    
    if response.status_code == 200:
        pdf_path = os.path.join(download_dir, f"{paper.entry_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return pdf_path, None
    else:
        return None, "Error downloading PDF"

def parse_and_create_db(pdf_paths: list):
    documents = []
    
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        docs = text_splitter.split_text(text)
        documents.extend(docs)
    
    embeddings = embedding_model.encode(documents)
    faiss_index = FAISS(embeddings)
    
    return documents, faiss_index

def query_papers(query: str, faiss_index, documents):
    chain = load_qa_chain(llm=llm, chain_type="refine", retriever=faiss_index)
    results = chain.run(query=query, documents=documents)
    
    return results

def related_papers(query: str, faiss_index, documents):
    related_titles = [doc['title'] for doc in documents]  # Assuming titles are stored
    return related_titles

# Streamlit interface
st.title("ArXiv Paper Query Assistant")

st.write("""
## Query Your Favorite ArXiv Papers!

- **Upload Papers**: Provide the arXiv links to the papers you want to query.
- **Ask Your Questions**: Enter your query to get precise answers from the uploaded papers.
""")

arxiv_links = st.text_area("Enter arXiv paper URLs (one per line):").splitlines()
query = st.text_input("Enter your query")

if st.button("Get Response"):
    if arxiv_links and query:
        pdf_paths = []
        for link in arxiv_links:
            pdf_path, error = download_arxiv_pdf(link)
            if error:
                st.error(error)
            else:
                pdf_paths.append(pdf_path)
        
        if pdf_paths:
            documents, faiss_index = parse_and_create_db(pdf_paths)
            response = query_papers(query, faiss_index, documents)
            related_titles = related_papers(query, faiss_index, documents)
            
            st.write("**Response:**")
            st.write(response)
            
            st.write("**Related Papers:**")
            for title in related_titles:
                st.write(title)
        else:
            st.error("No valid PDFs found.")
    else:
        st.warning("Please enter arXiv links and a query.")
