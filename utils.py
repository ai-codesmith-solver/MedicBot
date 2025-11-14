from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"))

def load_docs(path:str):
    if path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path=path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    
    return loader.load()

def get_retrive(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

