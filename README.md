# Bank Groq GenAI Chatbot

This project implements a chatbot powered by the Groq model/Ollama (for local) for answering questions related to bank policies and procedures. It utilizes multiple AI technologies, including Groq/Ollama, LangChain, and HuggingFace, to create a responsive and intelligent chatbot that can answer users' queries based on context extracted from PDF documents.

## Features

- **FAISS-based Similarity Search**: Efficient document retrieval using FAISS index for similar content based on user input.
- **Dynamic Conversation History**: Keeps track of user interactions and context for more coherent conversations.
- **PDF Document Parsing**: Extracts text from PDFs and uses the extracted data to inform the chatbot’s responses.
- **Streamlit Interface**: Provides an intuitive web interface for users to interact with the chatbot.
