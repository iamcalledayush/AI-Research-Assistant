import streamlit as st
import PyPDF2
import requests
import os
import tempfile
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Google Gemini 1.5 Flash model
def init_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.8,
        max_tokens=500,
        timeout=15,
        max_retries=3
    )

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Download PDF from arXiv link
def download_pdf(arxiv_url):
    pdf_url = arxiv_url.replace("abs", "pdf") + ".pdf"
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Failed to download PDF from {pdf_url}")
        return None

# Save PDF to a temporary file
def save_pdf(content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(content)
        return tmp_file.name

# Setup FAISS VectorStore for document retrieval
def create_faiss_index(texts):
    embeddings = HuggingFaceEmbeddings()
    faiss_index = FAISS.from_texts(texts, embeddings)
    return faiss_index

# Streamlit app
st.title("Your Own ArXiv Research Assistant")

# Store data in session state to persist across interactions
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'all_texts' not in st.session_state:
    st.session_state.all_texts = []

# Input arXiv links
arxiv_links = st.text_area("Enter the arXiv links (one per line):").splitlines()

if arxiv_links and st.button("Process Papers"):
    all_texts = []
    
    for link in arxiv_links:
        with st.spinner(f"Processing {link}..."):
            pdf_content = download_pdf(link)
            if pdf_content:
                pdf_file_path = save_pdf(pdf_content)
                with open(pdf_file_path, "rb") as pdf_file:
                    text = extract_text_from_pdf(pdf_file)
                    all_texts.append(text)
    
    if all_texts:
        # Create FAISS index with the text extracted from the papers
        st.write("Creating document retrieval index...")
        st.session_state.faiss_index = create_faiss_index(all_texts)
        st.session_state.all_texts = all_texts
        st.success("Index created successfully!")

# Only show the question input and retrieval system if the FAISS index exists
if st.session_state.faiss_index:
    # User question input
    user_question = st.text_input("I can help you do further research based on the uploaded papers. Ask your queries based on the uploaded documents:")

    if user_question:
        # Initialize the LLM and create a retrieval chain
        if 'llm' not in st.session_state:
            st.session_state.llm = init_llm("AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU")  # Use your API key
        
        qa_chain = load_qa_chain(st.session_state.llm, chain_type="stuff")
        retriever = st.session_state.faiss_index.as_retriever()
        chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

        # Answer the user's question using RAG
        with st.spinner("Generating answer..."):
            try:
                response = chain.run(user_question)
                st.write(f"**Answer:** {response}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
