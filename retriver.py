from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from utils import get_llm

llm=get_llm()


def create_retriver(vectore_store):
    retriver=MultiQueryRetriever.from_llm(
        retriever=vectore_store.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )

    

    return retriver