from flask import Flask, render_template, request
from utils import load_docs, get_retrive, get_llm
from vectore_store import create_vectore_store
from retriver import create_retriver
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from agnets import get_extra_context
from prompt import main_prompt


app = Flask(__name__)


str_parse = StrOutputParser()
llm = get_llm()

file_path = "source/Medical_book.pdf"

# DOCUMENT LOADER
print("📄 Loading medical documents... Please wait 🕒")
docs = load_docs(file_path)
print(f"Document Loded: {len(docs)}")

# VECTOR STORE 
print("\n🧠 Building vector store from medical embeddings... ⚙️")
vectore_store = create_vectore_store(docs)
print(f"Vectore Store Loded. ✅")

chat_history = []


# RAG FUNCTION 
def main_rag(query):

    retriver = create_retriver(vectore_store)

    chat_history.append({'user': query})

    parallel_chain = RunnableParallel({
        'context': retriver | RunnableLambda(get_retrive),
        'question': RunnablePassthrough(),
        'chat_history': RunnableLambda(lambda x: chat_history),
        'extra_context': RunnableLambda(get_extra_context)
    })

    main_chain = parallel_chain | main_prompt | llm | str_parse

    result = main_chain.invoke(query)

    chat_history.append({'bot': result})

    return result



# FLASK ROUTES
@app.route("/")
def main():
    return render_template("main.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_inp = request.form.get("text")
    if not user_inp:
        return "Please enter a message."

    # Run RAG pipeline instead of OpenAI API call
    bot_res = main_rag(user_inp)

    return bot_res



# RUN
if __name__ == "__main__":
    print("🚀 MedicBot RAG Flask Server Running...")
    app.run(debug=True)
