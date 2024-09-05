import os
import streamlit as st
import PyPDF2
import docx
from io import StringIO
from pyvis.network import Network
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up Google API Key directly in the code
GOOGLE_API_KEY = "AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU"

# Initialize the Gemini 1.5 Flash model
def init_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Using Gemini 1.5 Flash as per request
        api_key=api_key,
        temperature=0.8,  # Adjust temperature for creative responses
        max_tokens=500,    # Adjust max tokens for better performance
        timeout=15,        # Timeout after 15 seconds
        max_retries=3      # Retry up to 3 times in case of errors
    )

# Helper function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Helper function to extract text from text files or docx files
def extract_text_from_file(file):
    if file.type == "text/plain":
        return StringIO(file.getvalue().decode("utf-8")).read()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# Function to generate knowledge graph from AI output
def generate_knowledge_graph(concepts):
    # Initialize the knowledge graph
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

    # Add nodes and edges from the concepts
    for concept, related in concepts.items():
        net.add_node(concept, label=concept)
        for rel in related:
            net.add_node(rel, label=rel)
            net.add_edge(concept, rel)

    return net

# Streamlit interface
st.title("AI-Powered Knowledge Graph Generator")

# File upload or manual text input
uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "docx"])
manual_text = st.text_area("Or manually enter the text here")

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    else:
        document_text = extract_text_from_file(uploaded_file)
elif manual_text:
    document_text = manual_text
else:
    st.write("Please upload a file or enter text manually.")
    st.stop()

# Initialize LLM
llm = init_llm(GOOGLE_API_KEY)

# Option to create knowledge graph
if st.button("Generate Knowledge Graph"):
    with st.spinner("Extracting key concepts and relationships..."):
        # Generate the AI prompt for extracting key concepts
        prompt = (
            f"Extract the key concepts and their relationships from the following document. "
            f"Provide the concepts in a format suitable for building a knowledge graph. "
            f"Here is the document:\n\n{document_text}"
        )
        
        # Invoke the AI model
        response = llm.invoke(prompt)

        # Parse the AI's response (assuming it returns a JSON-like structure with key concepts)
        # Example format: {"Concept1": ["RelatedConcept1", "RelatedConcept2"], "Concept2": ["RelatedConcept3"]}
        try:
            concepts = eval(response.content)
        except:
            st.error("Error parsing the AI response. Please try again.")
            st.stop()

        # Generate and display the knowledge graph
        net = generate_knowledge_graph(concepts)
        net.show("knowledge_graph.html")

        # Embed the graph in the Streamlit app
        with open("knowledge_graph.html", "r", encoding="utf-8") as f:
            graph_html = f.read()
            components.html(graph_html, height=600)
