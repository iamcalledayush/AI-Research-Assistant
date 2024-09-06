import streamlit as st
import PyPDF2
import requests
import os
import tempfile
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

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
    # Convert plain text to Document objects required by LangChain FAISS index
    docs = [Document(page_content=text) for text in texts]
    faiss_index = FAISS.from_documents(docs, embeddings)
    return faiss_index

# Streamlit app
st.title("AI-Powered ArXiv and Document Research Assistant")

st.write(
    """
    This tool allows you to input arXiv links or upload PDFs, and then ask questions based on the contents of those documents.
    It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the documents and provide intelligent responses.
    """
)

# Store data in session state to persist across interactions
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'all_texts' not in st.session_state:
    st.session_state.all_texts = []
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'responses' not in st.session_state:
    st.session_state.responses = []  # Store responses across questions
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to reset session state
def reset_state():
    st.session_state.faiss_index = None
    st.session_state.all_texts = []
    st.session_state.llm = None
    st.session_state.responses = []  # Clear responses when resetting
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Reset memory

# Choose input method: either arXiv links or PDF upload
input_method = st.radio("Choose input method:", ("ArXiv Links", "Upload PDFs"), on_change=reset_state)

if input_method == "ArXiv Links":
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

elif input_method == "Upload PDFs":
    # Upload PDF documents
    uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])

    if uploaded_files and st.button("Process Uploaded PDFs"):
        all_texts = []
        
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                text = extract_text_from_pdf(uploaded_file)
                all_texts.append(text)
        
        if all_texts:
            # Create FAISS index with the text extracted from the uploaded PDFs
            st.write("Creating document retrieval index...")
            st.session_state.faiss_index = create_faiss_index(all_texts)
            st.session_state.all_texts = all_texts
            st.success("Index created successfully!")

# Function to separate LaTeX math and text, rendering LaTeX only in math mode
def render_response(response):
    # Find LaTeX enclosed in $...$ or $$...$$ and render only those as LaTeX math
    parts = re.split(r'(\$.*?\$|\$\$.*?\$\$)', response)
    
    for part in parts:
        # Handle inline math ($...$) or block math ($$...$$)
        if part.startswith("$") and part.endswith("$"):  # Math mode
            st.latex(part.strip("$"))
        else:  # Regular text
            st.write(part)

# Function to handle question submission and answer generation
def handle_question(user_question):
    if user_question:  # Prevent empty submissions
        # Initialize the LLM and create a retrieval chain
        if st.session_state.llm is None:
            st.session_state.llm = init_llm("AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU")  # Use your API key

        # Use FAISS similarity search for retrieving relevant documents
        retrieved_docs = st.session_state.faiss_index.similarity_search(user_question, k=3)  # Retrieve top 3 documents

        # Create a retrieval chain with the retrieved documents
        qa_chain = load_qa_chain(st.session_state.llm, chain_type="stuff")

        # Answer the user's question using RAG with memory
        with st.spinner("Generating answer..."):
            try:
                response = qa_chain.run(input_documents=retrieved_docs, question=user_question)
                # Append the new response to the list of responses
                st.session_state.responses.append({"question": user_question, "answer": response})
                return response  # Return response immediately to display it
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return None

# Chat interface setup
st.write("### Chat History")

# Display chat history (previous responses) at the top
if st.session_state.responses:
    for idx, res in enumerate(st.session_state.responses):
        st.write(f"**You**: {res['question']}")
        render_response(res['answer'])
        st.write("---")  # Divider between messages

# User question input, placed at the bottom like a chat interface
user_question = st.text_area(
    "Ask your question below:",
    placeholder="Ask a question... (Press Enter to submit, Shift+Enter for new line)",
    key="user_question"
)

# Button to trigger the response generation
if st.button("Send"):
    response = handle_question(user_question)
    if response:
        st.experimental_rerun()  # Refresh to display the new response at the top
