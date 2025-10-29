# -*- coding: utf-8 -*-

"""
For understanding RAG architecture : https://www.pinecone.io/learn/retrieval-augmented-generation/
RAG with LlamaIndex + FAISS + Local LLM (Ollama)
This script demonstrates a fully open-source Retrieval-Augmented Generation (RAG) pipeline
using LlamaIndex for indexing and querying, FAISS for vector storage, and a local LLM via Ollama.
"""
# Install dependencies (run in terminal once)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# 🦙 Fully Open-Source RAG pipeline using LlamaIndex + FAISS + local LLM


# 1️⃣ Load your custom data (directory of .txt, .pdf, .md, etc.)
data_path = "path_to_your_documents"
documents = SimpleDirectoryReader(data_path).load_data()


# 2️⃣ Create a local embedding model (Hugging Face)
# Choose any small embedding model that works offline
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 3️⃣ Setup FAISS vector store (fully local)
embedding_dim = 384  # for all-MiniLM-L6-v2
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=faiss_store)


# 4️⃣ Create an index (store embeddings locally)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)


# 5️⃣ Configure your local LLM (via Ollama)
# Make sure Ollama is installed and running (https://ollama.ai)
# You can use: `ollama pull llama3`  or  `ollama pull mistral`
llm = Ollama(model="llama3")  # or "mistral", "phi3", etc.


# 6️⃣ Create the query engine (retrieve + reason locally)
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3  # number of most relevant chunks to retrieve
)


# 7️⃣ Function to query your data locally
def generate_response(query: str):
    """RAG query using local embeddings + local LLM."""
    response = query_engine.query(query)
    return str(response)


# Example usage
if __name__ == "__main__":
    user_query = "How do I reset my device?"
    print(generate_response(user_query))