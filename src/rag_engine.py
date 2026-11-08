import ollama
from typing import List, Dict

class RAGEngine:
    """Orchestrates the retrieval and generation process."""
    
    def __init__(self, vector_store, llm_model: str = "qwen2.5vl:3b"):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.client = ollama.Client(host="http://127.0.0.1:11434")

    def generate_answer(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Performs RAG: retrieves context and generates an answer using the LLM."""
        
        # 1. Retrieve context
        retrieved_chunks = self.vector_store.query(query)
        context = "\n\n".join(retrieved_chunks)

        # 2. Construct System Prompt
        system_prompt = f"""
You are a personal AI assistant.

Answer questions based on document context. 

DOCUMENT CONTEXT:
{context}
"""

        # 3. Build conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        # 4. Generate answer
        response = self.client.chat(
            model=self.llm_model,
            messages=messages,
            options={"temperature": 0.2}
        )

        answer = response["message"]["content"]
        return answer
