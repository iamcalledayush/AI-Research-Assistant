import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
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
    model="gemini-1.5-flash",  # Use Gemini 1.5 Flash model here
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
    
    # Extract the paper ID from the arXiv URL
    paper_id = arxiv_url.split('/')[-1]
    
    # Construct the direct PDF URL
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    response = requests.get(pdf_url)
    
    if response.status_code == 200:
        pdf_path = os.path.join(download_dir, f"{paper_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return pdf_path, None
    else:
        return None, f"Error downloading PDF from {pdf_url}"

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
    
    # Create FAISS index
    embeddings = embedding_model.encode(documents)  # Using SentenceTransformer directly
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Map documents to the FAISS index
    docstore = InMemoryDocstore({i: Document(page_content=doc) for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: i for i in range(len(documents))}
    
    # Initialize FAISS with the embedding function
    faiss_index = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model.encode  # Pass the embedding function here
    )
    
    return documents, faiss_index

def query_papers_separately(query: str, faiss_index, documents):
    # Iterate over each document and explicitly process them individually in a single call
    results = []
    
    for i, doc in enumerate(documents):
        # Create a specific prompt for each document
        paper_prompt = f"Paper {i+1} Content:\n{doc.page_content}\n\nPlease focus only on explaining this paper's content and do not mention or explain any related papers that might be cited or referenced within it."
        
        # Invoke the LLM for this specific paper
        result = llm.invoke(f"{paper_prompt}\n\nQuestion: {query}")
        
        # Collect the response for each paper
        results.append(f"Response for Paper {i+1}:\n{result.content}\n")
    
    # Concatenate all responses
    return "\n".join(results)

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
            pdf_paths = []
            for link in arxiv_links:
                pdf_path, error = download_arxiv_pdf(link)
                if error:
                    st.error(error)
                else:
                    pdf_paths.append(pdf_path)
            
            if pdf_paths:
                documents, faiss_index = parse_and_create_db(pdf_paths)
                
                # Perform LLM calls with clear separation for each paper
                combined_response = query_papers_separately(query, faiss_index, documents)
                
                # Display the results
                st.write("**Combined Papers Response:**")
                st.write(combined_response)
            else:
                st.error("No valid PDFs found.")
    else:
        st.warning("Please enter arXiv links and a query.")
