import os
import asyncio
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure all necessary imports are at the very top
from pinecone import Pinecone
import google.generativeai as genai
from google.generativeai import types

# --- STREAMLIT PAGE CONFIG MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Pinecone-Gemini RAG Chatbot", layout="wide")
# ---------------------------------------------------------------------

# Load environment variables (should be early, after imports)
load_dotenv()

# Configure Gemini API key and model
# These can be defined now that genai is imported
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
RESPONSE_MODEL = "gemini-2.0-flash-lite"
EMBEDDING_MODEL = "models/text-embedding-004"

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Extract cloud and region from pinecone_env for error message clarity
try:
    cloud, region = pinecone_env.split("-", 1)
except AttributeError:
    st.error("Error: PINECONE_ENV environment variable is not set correctly. It should be in the format 'cloud-region' (e.g., 'aws-us-east-1').")
    st.stop()
except ValueError:
    st.error("Error: PINECONE_ENV environment variable is not in the expected 'cloud-region' format.")
    st.stop()


# Initialize Pinecone client and check index existence
try:
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

    existing_index_names = {idx['name'] for idx in pc.list_indexes()}

    if INDEX_NAME not in existing_index_names:
        st.error(f"Error: Pinecone index '{INDEX_NAME}' not found. Please ensure it is created in your Pinecone dashboard with cloud='{cloud}' and region='{region}'.")
        st.stop()
    else:
        st.success(f"Connecting to existing Pinecone index: {INDEX_NAME}.")
        pinecone_index = pc.Index(INDEX_NAME)
        st.success(f"Successfully connected to Pinecone index: {INDEX_NAME}.")
except Exception as e:
    st.error(f"Failed to initialize Pinecone client or connect to index: {e}")
    st.stop()


async def get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    """Use Gemini embedding API."""
    try:
        embedding_resp = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        return embedding_resp["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return [0.0] * 768

async def retrieve_context_from_pinecone(query_embedding: List[float], top_k: int = 5) -> List[str]:
    """Retrieve relevant text chunks from Pinecone based on query embedding."""
    try:
        query_results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        contexts = [match.metadata['content'] for match in query_results.matches if 'content' in match.metadata]
        return contexts
    except Exception as e:
        st.error(f"Pinecone retrieval error: {e}")
        return []

async def answer_question_with_llm(question: str, contexts: List[str]) -> str:
    """Answer a user question using Gemini, given retrieved contexts."""
    if not contexts:
        return "I don't have enough information to answer that question from the available documents."

    context_str = "\n\n".join(contexts)

    prompt = (
        "You are an assistant that answers questions based on the provided context. "
        "If the answer is not in the context, state that you don't have enough information.\n\n"
        "Context:\n"
        f"{context_str}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )

    try:
        model = genai.GenerativeModel(RESPONSE_MODEL)
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini LLM generation error: {e}")
        return "An error occurred while generating the response."


# --- Streamlit UI ---
st.title("RCB Website Information Assistant") # Title changed to "Rcb Web Page Rag"

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            query_embedding = asyncio.run(get_embedding(prompt, task_type="RETRIEVAL_QUERY"))
            contexts = asyncio.run(retrieve_context_from_pinecone(query_embedding))
            full_response = asyncio.run(answer_question_with_llm(prompt, contexts))

            st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})