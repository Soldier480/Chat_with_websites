import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
HF_TOKEN=os.getenv('HF_TOKEN')
CHROMA_PATH = "chroma"

if not openai_api_key:
    st.error("OPENAI API key is missing. Please check your .env file.")
    st.stop()
if not HF_TOKEN:
    st.error("HF TOKEN key is missing. Please check your .env file.")
    st.stop()

def get_response(user_input):
    retriever_chain=get_context_retriever_chain(st.session_state.vector_store)
    conversatin_rag_chain=get_conversational_rag_chain(retriever_chain)
    response=conversatin_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
    return response['answer']
def get_vectorstore_from_url(url):
    # get the text in document form
  try:
        loader = WebBaseLoader(url)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
         # Use SentenceTransformers for embedding
        model =  HuggingFaceEmbeddings(
         model_name = "sentence-transformers/all-MiniLM-L6-v2")

        # Create a Chroma vector store with the embedding function
        vector_store = Chroma.from_documents(
        documents=document_chunks, 
        embedding=model, 
        persist_directory=CHROMA_PATH  )
        vector_store.persist()
        
        return vector_store
  except Exception as e:
        st.error(f"An error occurred while processing the URL: {e}")
        return None
def get_context_retriever_chain(vector_store):
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    llm=HuggingFaceEndpoint(repo_id=repo_id, token=HF_TOKEN)
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
     
     repo_id="mistralai/Mistral-7B-Instruct-v0.2"

     llm=HuggingFaceEndpoint(repo_id=repo_id, token=HF_TOKEN)
     
     prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
     stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
     return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Website URL")

if website_url is None or website_url=="":
    st.info("please enter a website url")

else:
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= [
        AIMessage(content="Hello, I am an AI assistant. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
       
    

    #user_input
    user_query=st.chat_input("Type your message here...")
    if user_query is not None and user_query!="":
        response=get_response(user_query)       
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        
    #  conversation 
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)