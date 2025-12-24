# Umbrella Onboarding Assistant
A Streamlit-based onboarding assistant that answers employee questions using a combination of:
- employee profile context (synthetic employee records for local dev/testing), and
- internal policy/regulations retrieved from a PDF via a local vector store (RAG).

The goal is to provide fast, grounded answers to common onboarding questions like time off, safety protocols, security procedures, and benefits.

## Features
- **Personalized context**: Uses an employee profile (name, role, department, location, skills) to tailor answers.
- **Policy-aware responses (RAG)**: Retrieves relevant policy excerpts from an internal regulations PDF and uses them to ground responses.
- **Streaming responses**: Answers stream in real time in the UI.
- **Session persistence**: Employee profile + chat history are cached for the session.

## Tech Stack
- **Streamlit** for the UI
- **LangChain** for orchestration (retrieval + prompting)
- **Vector store** (FAISS or similar) for local semantic search
- **Embeddings + LLM API** (configured via environment variables)

## Project Structure (typical)
- `app.py` (or `streamlit_app.py`): Streamlit entrypoint
- `synthetic_data/`: module for generating employee profiles
- `data/` or `assets/`: PDF policies/regulations (local)
- `vectorstore/` (or similar): persisted embeddings index (generated)