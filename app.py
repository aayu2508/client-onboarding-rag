from data.employees import generate_employee_data
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import logging
from assistant import Assistant
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from langchain_groq import ChatGroq
from gui import AssistantGUI

if __name__ == "__main__":
    load_dotenv(".env.local")
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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
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
    vector_store = init_vector_store("data/umbrella_corp_policies.pdf")

    if "customer" not in st.session_state:
        st.session_state.customer = user_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

    llm = ChatGroq(model="llama-3.3-70b-versatile")
    assistant = Assistant(
            system_prompt=SYSTEM_PROMPT,
            llm=llm,
            message_history=st.session_state.messages,
            user_information=st.session_state.customer,
            vector_store=vector_store,
        )    

    gui = AssistantGUI(assistant)
    gui.render()