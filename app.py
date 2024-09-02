import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain.docstore import InMemoryDocstore
import faiss
from langchain.docstore.document import Document

# Set up Google API Key directly in the code
GOOGLE_API_KEY = "AIzaSyCSOt-RM3M-SsEQObh5ZBe-XwDK36oD3lM"

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Direct use of SentenceTransformer
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Use Gemini 1.5 Pro model here
    api_key=GOOGLE_API_KEY,  # Pass the API key here
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Agent 1: Document Parsing
def download_and_parse_papers(arxiv_urls):
    parsed_papers = []
    for url in arxiv_urls:
        paper_id = url.split('/')[-1]
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        response = requests.get(pdf_url)
        if response.status_code == 200:
            text = extract_text_from_pdf(response.content)
            parsed_papers.append(text)
        else:
            st.error(f"Error downloading PDF from {pdf_url}")
    return parsed_papers

def extract_text_from_pdf(pdf_content):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_content)
    reader = PdfReader("temp.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Agent 2: Summarization
def summarize_paper(paper_text):
    prompt = (
        "The following is the content of a research paper. "
        "Please summarize the content without referencing or explaining any related papers:\n\n"
        + paper_text
    )
    summary = llm.invoke(prompt)
    return summary.content

# Agent 3: Response Integration
def integrate_responses(summaries):
    integrated_response = "\n\n".join(f"Paper {i+1} Summary:\n{summary}" for i, summary in enumerate(summaries))
    return integrated_response

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
        with st.spinner('Processing your request...'):
            # Agent 1: Parse papers
            parsed_papers = download_and_parse_papers(arxiv_links)
            
            # Agent 2: Summarize papers
            summaries = [summarize_paper(paper) for paper in parsed_papers]
            
            # Agent 3: Integrate responses
            final_response = integrate_responses(summaries)
            
            # Display the results
            st.write("**Combined Papers Response:**")
            st.write(final_response)
    else:
        st.warning("Please enter arXiv links and a query.")
