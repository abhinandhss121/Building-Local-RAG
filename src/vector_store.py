import ollama
import chromadb
from typing import List, Dict, Any

class VectorStore:
    """Manages embeddings and the vector database using Ollama and ChromaDB."""
    
    def __init__(self, db_path: str = "./chroma_db", embedding_model: str = "nomic-embed-text"):
        self.embedding_model = embedding_model
        self.client = ollama.Client(host="http://127.0.0.1:11434")
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text chunks."""
        response = self.client.embed(
            model=self.embedding_model,
            input=texts
        )
        return response["embeddings"]

    def add_documents(self, chunks: List[str], metadata_source: str):
        """Embeds and stores document chunks in ChromaDB."""
        embeddings = self.generate_embeddings(chunks)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": metadata_source, "chunk_index": i} for i in range(len(chunks))]
        
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_text: str, top_k: int = 3) -> List[str]:
        """Retrieves the most relevant chunks for a given query."""
        query_embedding = self.generate_embeddings([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results["documents"][0]
