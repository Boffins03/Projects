from langchain_community.embeddings import HuggingFaceEmbeddings
from load_json_file import load_json_file
from langchain_community.vectorstores import FAISS

# Load documents from JSON file
documents = load_json_file("project_1_publications.json")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create or load vector store
vectorstore = FAISS.from_documents(documents, embeddings)

vectorstore.save_local("faiss_index")

# Load the vector store
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Example usage:
query = "How to add memory to the RAG?"
docs = vectorstore.similarity_search(query, k=2)

print("\nTop matches:")
for d in docs:
    print("----")
    content = d.page_content
    print(content) 