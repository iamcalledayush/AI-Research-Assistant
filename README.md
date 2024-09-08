# AI-Powered Research Paper Assistant

## Overview
The AI-Powered Research Paper Assistant is a robust tool that allows users to upload multiple research papers (PDF format) or provide ArXiv links, enabling them to query the content and receive accurate, context-aware responses. The tool uses advanced document retrieval techniques combined with the power of Google Gemini 1.5 Flash to generate detailed, insightful answers to user questions. It leverages Retrieval-Augmented Generation (RAG) to retrieve relevant context from the uploaded documents and provides intelligent, context-aware responses. 

## Key Features
- **Multi-Document Support:** Users can upload multiple PDFs or enter multiple ArXiv links.
- **Context-Aware Question Answering:** Provides intelligent, memory-enhanced responses that can handle follow-up questions while keeping the context intact.
- **Similarity Search with FAISS:** Utilizes Meta’s FAISS for efficient similarity search across documents to find the most relevant information.
- **Interactive User Interface:** Built with Streamlit, providing a simple, intuitive interface for users to interact with the assistant.
- **Memory-Integrated Chat:** The system uses memory capabilities via LangChain to ensure follow-up questions are accurately answered by referring to prior interactions.

## Technologies Used
- **LangChain:** Provides an efficient framework for managing large language models (LLMs) and chains, enabling dynamic question-answering based on document content.
- **Google Gemini 1.5 Flash LLM:** Used in the backend to process user questions and generate responses based on the context provided by the documents.
- **FAISS (Facebook AI Similarity Search):** A library for efficient similarity search, enabling fast document retrieval based on user queries.
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval and LLM capabilities to provide context-rich answers.
- **Streamlit:** Used for building the frontend, offering an interactive, easy-to-use web interface.
- **MLOps:** The entire system is fully end-to-end deployed using modern MLOps practices, ensuring scalability, performance, and robustness.

## How It Works
1. **Document Upload:** Users can either upload PDF files or provide ArXiv links.
2. **Document Parsing:** The system extracts text from the uploaded PDFs or ArXiv papers.
3. **Index Creation:** FAISS is used to create a vector store of the document content, allowing for fast similarity-based search.
4. **Question Answering:** Users can input research questions, and the system searches for relevant sections in the documents, generating responses through Google Gemini 1.5 Flash. Memory is utilized to ensure the system understands the context of follow-up questions.
5. **Context-Aware Conversations:** The assistant retains conversation history, allowing users to ask follow-up questions and receive answers with the context from previous interactions.

## Usage Instructions
1. Upload research papers (PDF) or provide ArXiv links.
2. Ask your research-related questions based on the content of the uploaded documents.
3. The assistant will retrieve relevant information and provide an accurate, context-aware answer.
4. You can ask follow-up questions, and the assistant will keep track of the conversation history for more insightful responses.
