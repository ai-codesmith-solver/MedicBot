from utils import load_docs,get_retrive,get_llm
from vectore_store import create_vectore_store
from retriver import create_retriver
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from agnets import get_extra_context
from prompt import main_prompt


str_parse=StrOutputParser()

llm=get_llm()

file_path="source/Medical_book.pdf"

# document loader
docs=load_docs(file_path)

# chunking + vectore DB
vectore_store=create_vectore_store(docs)


chat_history=[]

def main_rag(query):

    # retriver
    retriver=create_retriver(vectore_store)

    chat_history.append({'user':query})

    parallel_chain=RunnableParallel({
        'context': retriver | RunnableLambda(get_retrive),
        'question': RunnablePassthrough(),
        'chat_history':RunnableLambda(lambda x: chat_history),
        'extra_context': RunnableLambda(get_extra_context)
    })

    main_chain= parallel_chain | main_prompt | llm | str_parse

    result=main_chain.invoke(query)
    chat_history.append({'bot':result})

    return result


if __name__=="__main__":
    while True:
        query = input("\n🔎 Ask a question (or type 'exit' to quit): ").strip()

        if query.lower() == "exit":
            break
        print("\n⏳ Generating answer...")

        result=main_rag(query)

        if result :
            print("\n💡 Final Answer:")
            print(result)



