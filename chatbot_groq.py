# https://github.com/leoneversberg/llm-chatbot-rag/blob/main/src/rag_util.py
# https://github.com/AlaGrine/RAG_chatabot_with_Langchain/blob/main/RAG_app.py
# https://medium.com/thedeephub/rag-chatbot-powered-by-langchain-openai-google-generative-ai-and-hugging-face-apis-6a9b9d7d59db
# https://github.com/mycloudtutorials/generative-ai-demos/blob/master/bedrock-chat-with-pdf/User/app.py

import openai
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Path to store the FAISS index
FAISS_INDEX_PATH = "faiss_index"
CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Chatbot")
)

def load_or_create_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L12-v2"

    # Check if the FAISS index already exists
    if os.path.exists(FAISS_INDEX_PATH):
        # Load existing FAISS index and embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": "cpu"}
        )
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True) #TODO do not use it for untrusted files
        print("Loaded existing FAISS index.")
    else:
        # Create new FAISS index if not found
        raw_text = ''
        source_documents = []  # This will hold the source documents
        
        # Loop through all the files in the folder
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(CACHE_DIR, filename)
                doc_reader = PdfReader(pdf_path)
                source_documents.append(pdf_path)
                
                # Loop through each page of the PDF and extract text
                for i, page in enumerate(doc_reader.pages):
                    text = page.extract_text()
                    if text:
                        raw_text += text
                        source_documents.append(pdf_path)  # Append the document path
        
        # Splitting up the text into smaller chunks for indexing
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200, #striding over the text
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": "cpu"},
        )

        db = FAISS.from_texts(texts, embeddings)

        # Save the FAISS index to disk for future use
        db.save_local(FAISS_INDEX_PATH)
        print("Created and saved new FAISS index.")
    
    return db

def retrieve_context(user_input):
    # Load or create the FAISS embeddings
    db = load_or_create_embeddings()

    # Perform similarity search
    docs = db.similarity_search(user_input, k=1)
    #print('in retrieve_context, user_input: ', user_input)
    #print('In retrieve_context, docs: ', docs)
    return docs 


def generate_response(user_input):
    # Step 1: Retrieve relevant context via similarity search
    relevant_context = retrieve_context(user_input)  
    
    # Combine the relevant context with the conversation history
    if relevant_context: 
        context = "\n".join([doc.page_content for doc in relevant_context])  # Extract text from each Document object
    else:
        context = ''

    if not relevant_context:
        return "I'm sorry, I don't know the answer to that."    
    
    # Build the conversation prompt by joining the history and the relevant context
    instructions = f"""You are a friendly and helpful assistant. 
    Your task is to assist users with questions about Bank policies and procedures only. You are given context for the same. 
    Please use the given context to provide concise answer to the user's question.
    Don't try to make up an answer if it not provided in the context.
    If user asks question about topics not related to Bank policies and procedures, just say "I'm sorry, I don't know the answer to that." 
    If you don't know the answer, just say "I'm sorry, I don't know the answer to that". 
    Do not try to make up an answer if not given in the context.
    However, for normal greetings, like "Hi", "How are you" etc, you can respond politely, concisely and appropriately. 
    For normal greetings, keep your answer short, not more than a few words.
    For questions related to Bank policies and procedures, respond in maximum 4-5 sentences.
    """

    # instructions = """Please use the given context to provide concise answer to the question
    # If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    
    prompt = instructions + "\n".join(st.session_state.history) + f"\nUser: {user_input}\nContext: {context}\nBot:"
    print('-------------------------------------------------------')
    print('user_input: ', user_input)
    print('context: ', context)
    try:
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        message = completion.choices[0].message.content
        
        # Only append source information for queries that require documents
        # if relevant_context:
        #     source_info = "\n".join([f"Source: {doc}" for doc in source_documents])
        #     response = f"{message}\n\n{source_info}"  # Append source document info
        # else:
        response = message  # For greetings or non-document-based responses, no source info
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI layout
st.set_page_config(page_title="Bank Groq GenAI Chatbot", layout="wide")

# Custom title and description
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Bank Groq GenAI Chatbot</h1>
    <p style='text-align: center;'>Chat with an AI assistant powered by Groq model. Ask questions, get creative, or have fun conversations!</p>
    <hr style="border:1px solid gray;">
""", unsafe_allow_html=True)

# Create a text input box with a submit button
user_input = st.text_input("Type your message here and press Enter:", "", key="input", help="Type your message here and press Enter.")

# Handle user interaction and show the conversation

if user_input.strip() != "":
    # Add user input to history
    st.session_state.history.append(f"User: {user_input}")
    
    # Get the model's response
    response = generate_response(user_input)
    
    # Add bot's response to history
    st.session_state.history.append(f"Bot: {response}")

# Display the chat history
st.markdown("<h3 style='color: #4CAF50;'>Conversation History:</h3>", unsafe_allow_html=True)
for message in reversed(st.session_state.history):
    if message.startswith("User:"):
        st.markdown(f"<div style='background-color: #f1f1f1; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<strong>You:</strong> {message[6:]}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<strong>Bot:</strong> {message[5:]}</div>", unsafe_allow_html=True)

# Button to clear the conversation history
if st.button("Clear Chat History"):
    st.session_state.history = []
    st.experimental_rerun()

# Footer with some styling
st.markdown("""
    <hr style="border:1px solid gray;">
    <p style="text-align: center; color: gray;">Powered by Groq | Developed with Streamlit</p>
""", unsafe_allow_html=True)
