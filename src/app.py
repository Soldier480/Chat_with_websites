import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)


# langchain >= 1.0 moved these chains to langchain_classic
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # change if unavailable on HF
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if not HF_TOKEN:
    st.error("HF token is missing. Please check your .env file.")
    st.stop()

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
        max_retries=2,
        timeout=30,
    )


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_vectorstore_from_url(url):
    try:
        loader = WebBaseLoader(url)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(document)

        # in-memory store: no stale/duplicate data between runs
        return Chroma.from_documents(documents=chunks, embedding=get_embeddings())
    except Exception as e:
        st.error(f"An error occurred while processing the URL: {e}")
        return None


def get_context_retriever_chain(vector_store):
    llm = get_llm()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up "
                 "in order to get information relevant to the conversation"),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(retriever_chain):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)
    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
    })
    return response["answer"]


# ---------- app config ----------
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if not website_url:
    st.info("Please enter a website URL")
else:
    # session_state persists values across Streamlit reruns
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am an AI assistant. How can I help you?")
        ]

    # (re)build the vector store when the URL is new or has changed
    if st.session_state.get("loaded_url") != website_url:
        with st.spinner("Loading website..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
        st.session_state.loaded_url = website_url
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am an AI assistant. How can I help you?")
        ]

    if st.session_state.vector_store is None:
        st.stop()

    user_query = st.chat_input("Type your message here...")
    if user_query:
        with st.spinner("Thinking..."):
            try:
                response = get_response(user_query)
            except Exception as e:
                response = f"Error while generating response: {e}"
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.write(message.content)