# --- Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

from load_json_file import load_json_file
from embedding import embedding
from model import load_model

# All community integrations (vectorstores, embeddings, loaders, LLMs)
from langchain_community.vectorstores import FAISS

# Core modules (prompts, chains)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Steps ---

# 1. Load documents
docs = load_json_file("project_1_publications.json")

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# 3. Create embeddings and vectorstore
embeddings = embedding()
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Load your language model (custom load_model() function)
llm = load_model()

# 5. Define chat prompt
prompt = ChatPromptTemplate.from_template("""
You are a strict assistant. 
Answer using only the given context. 
If the question is unrelated, respond:
"I’m sorry — I only know about the documents."

Context:
{context}

User question:
{input}

Answer:
""")

# 6. Create the document-combining and retrieval chains
combine_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_chain)

# 7. Chat loop
def chat():
    print("🔹 Chatbot ready! Type 'exit' or 'quit' to stop.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            print(" Goodbye!")
            break
        out = retrieval_chain.invoke({"input": q})
        # Handle both dict and string outputs
        if isinstance(out, dict):
            print("Bot:", out.get("output") or out.get("result") or out)
        else:
            print("Bot:", out)

if __name__ == "__main__":
    chat()
