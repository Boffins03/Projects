from langchain_community.embeddings import HuggingFaceEmbeddings

def embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



if __name__ == "__main__":
    embeddings = embedding()

    # Example usage:
    query = "What is the capital of France?"
    embedding = embeddings.embed_query(query)

    print(embedding)