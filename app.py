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

# Set up the two Google API Keys directly in the code
SUMMARY_API_KEY = "AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU"
FINAL_RESPONSE_API_KEY = "AIzaSyCSOt-RM3M-SsEQObh5ZBe-XwDK36oD3lM"

# Initialize components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Direct use of SentenceTransformer

def get_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  # Use Gemini 1.5 Pro model here
        api_key=api_key,  # Pass the respective API key
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

summary_llm = get_llm(SUMMARY_API_KEY)
final_response_llm = get_llm(FINAL_RESPONSE_API_KEY)

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
        documents.extend([Document(page_content=doc) for doc in docs])
    
    # Create FAISS index
    embeddings = embedding_model.encode([doc.page_content for doc in documents])  # Use the page_content
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Map documents to the FAISS index
    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: i for i in range(len(documents))}
    
    # Initialize FAISS with the embedding function
    faiss_index = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model.encode  # Pass the embedding function here
    )
    
    return documents, faiss_index

def summarize_paper(paper_content):
    # Create a prompt for generating a medium-length summary
    prompt = (
        f"Summarize the following paper content in a medium-length summary, focusing on the main points, and excluding any related papers:\n\n"
        f"{paper_content}"
    )
    
    try:
        summary = summary_llm.invoke(prompt)
        return summary.content
    except Exception as e:
        st.error(f"Error summarizing paper: {str(e)}")
        return None

def query_papers_combined(query: str, summaries: list):
    # Combine all summaries into a single prompt with clear separation
    combined_summaries = "\n\n".join([f"Paper {i+1} Summary:\n{summary}" for i, summary in enumerate(summaries)])
    
    # Create a prompt that instructs the LLM to focus only on the given summaries' content
    prompt = (
        "The following are summaries of multiple research papers. "
        "Each summary is separated and labeled. Please focus solely on these summaries "
        "and answer the following question based on the summaries provided:\n\n"
        + combined_summaries
        + "\n\nQuestion: "
        + query
    )
    
    try:
        final_answer = final_response_llm.invoke(prompt)
        return final_answer.content
    except Exception as e:
        st.error(f"Error generating final response: {str(e)}")
        return None

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
                
                # Summarize each paper individually
                summaries = []
                for doc in documents:
                    summary = summarize_paper(doc.page_content)
                    if summary:
                        summaries.append(summary)
                
                if summaries:
                    # Perform a combined LLM call using the final response API key
                    combined_response = query_papers_combined(query, summaries)
                    
                    # Display the results
                    if combined_response:
                        st.write("**Combined Papers Response:**")
                        st.write(combined_response)
                    else:
                        st.error("Failed to generate a combined response.")
                else:
                    st.error("Failed to generate summaries for the papers.")
            else:
                st.error("No valid PDFs found.")
    else:
        st.warning("Please enter arXiv links and a query.")
