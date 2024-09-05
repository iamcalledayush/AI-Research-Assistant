import os
import streamlit as st
import PyPDF2
import docx
from io import StringIO
from pptx import Presentation
from pptx.util import Inches
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


GOOGLE_API_KEY = "AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU"

# Initialize the Gemini 1.5 Pro model
def init_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.8,  # Adjust temperature for creative responses
        max_tokens=300,    # Adjust max tokens for better performance
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

# Function to create a more detailed PowerPoint from text
def create_ppt_from_text(text):
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]  # Use a layout with a title and content

    # Split the text into sections
    sections = text.split("\n\n")  # Assuming sections are separated by double newlines
    section_limit = 5  # Limit sections per slide to avoid too much text on one slide

    for i, section in enumerate(sections):
        if i % section_limit == 0:  # Start a new slide every few sections
            slide = prs.slides.add_slide(slide_layout)
            title = slide.shapes.title
            title.text = f"Slide {i // section_limit + 1}"

        content = slide.shapes.placeholders[1]
        content.text += "\n- " + section.strip()  # Add each section as a bullet point

    # Save PowerPoint
    ppt_filename = "detailed_presentation.pptx"
    prs.save(ppt_filename)
    return ppt_filename

# Streamlit interface
st.title("Document Processing App")

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

# Let the user choose the action
option = st.selectbox("Choose what you want to do with the text:", 
                      ["Create Summary", "Q&A Chatbot", "Create PPT from Text"])

# Initialize LLM
llm = init_llm(GOOGLE_API_KEY)

# Handle the user's choice
if option == "Create Summary":
    with st.spinner("Generating summary..."):
        summary_prompt = f"Summarize the following text: {document_text}"
        response = llm.invoke(summary_prompt)
        st.write("### Summary:")
        st.write(response.content)

elif option == "Q&A Chatbot":
    st.write("Ask questions about the document.")
    question = st.text_input("Enter your question:")
    
    if st.button("Ask"):
        with st.spinner("Fetching answer..."):
            qa_prompt = f"Based on this document: {document_text}\nAnswer the question: {question}"
            response = llm.invoke(qa_prompt)
            st.write("### Answer:")
            st.write(response.content)

elif option == "Create PPT from Text":
    with st.spinner("Creating detailed PowerPoint..."):
        ppt_file = create_ppt_from_text(document_text)
        with open(ppt_file, "rb") as f:
            st.download_button("Download PPT", f, file_name="detailed_presentation.pptx")
