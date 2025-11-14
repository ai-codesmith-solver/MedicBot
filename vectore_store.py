from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Pinecone Vectore DB Initialization
pinecone_api_key=os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Creating Pinecone Index
index_name = "medicbotdb"  

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

def create_vectore_store(docs):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )

    if not docs:
        raise ValueError("❌ No documents provided to create vector store.")

    texts = []

    # 🧠 Normalize to list of strings
    for item in docs:
        if isinstance(item, Document):
            texts.append(item.page_content)
        elif isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and "page_content" in item:
            texts.append(item["page_content"])
        else:
            print(f"⚠️ Skipping unsupported item: {type(item)}")

    # ✅ Double-check
    if not texts:
        raise ValueError("❌ No valid text found in provided documents.")
    
    chunks=splitter.create_documents(texts)

    #-----------------------------------------------
    #FAISS Vectore DB
    #------------------------------------------------

    # vectore_store=FAISS.from_documents(
    #     documents=chunks,
    #     embedding=embedding_model
    # )

    #-----------------------------------------------
    #Pinecone Vectore DB Loaded
    #------------------------------------------------

    # vectore_store=PineconeVectorStore.from_documents(
    #     documents=chunks,
    #     index_name=index_name,
    #     embedding=embedding_model
    # )

    #Conecting with exeisting pinecone db with the help of index
    vectore_store=PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )

    return vectore_store