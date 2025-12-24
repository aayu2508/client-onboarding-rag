from data.employees import generate_employee_data
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import logging
# from assistant import Assistant
# from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from langchain_groq import ChatGroq
# from gui import AssistantGUI

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    st.set_page_config(page_title="Umbrella Onboarding", page_icon=":umbrella:", layout="wide")

    # Generate employee data for a single user to send to our assistant
    @st.cache_data(ttl=3600, show_spinner="Loading employee data...")
    def get_user_data():
        return generate_employee_data(1)[0]

    # Initialize vector store for PDFs (embeddings created using OpenAI)
    @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
    def init_vector_store(pdfs_path):
        try:
            loader = PyPDFLoader(pdfs_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()

            # Persistent path for vector store
            persistent_path = ".data/vectorstore"
            vector_store = Chroma.from_documents(
                splits,
                embeddings,
                persist_directory=persistent_path
            )
            return vector_store
        except Exception as e:
            logging.error(f"Error initializing vector store: {e}")
            st.error(f"Error initializing vector store: {e}")
            return None
    
    user_data = get_user_data()
    vector_store = init_vector_store("data/umbrella_onboarding_guide.pdf")
    
    