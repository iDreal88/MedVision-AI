import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, knowledge_base_path, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.knowledge_base_path = knowledge_base_path
        self.chunks = []
        self.index = None
        self._load_and_index()

    def _load_and_index(self):
        if not os.path.exists(self.knowledge_base_path):
            print(f"Knowledge base not found at {self.knowledge_base_path}")
            return

        with open(self.knowledge_base_path, 'r') as f:
            content = f.read()

        # Simple chunking by sections (double newline)
        raw_chunks = content.split('\n\n')
        self.chunks = [c.strip() for c in raw_chunks if c.strip()]

        if not self.chunks:
            print("No content found in knowledge base.")
            return

        # Create embeddings
        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]

        # Use FAISS for indexing
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Indexed {len(self.chunks)} chunks from knowledge base.")

    def search(self, query, top_k=2):
        if self.index is None:
            return "No knowledge base indexed."

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)

        results = [self.chunks[idx] for idx in indices[0] if idx != -1]
        return "\n\n".join(results)

if __name__ == "__main__":
    # Test the engine
    kb_path = 'knowledge_base.md'
    if os.path.exists(kb_path):
        engine = RAGEngine(kb_path)
        test_query = "What are the characteristics of malignant tumors?"
        print(f"\nQuery: {test_query}")
        print("-" * 30)
        print(engine.search(test_query))
    else:
        print(f"Please ensure {kb_path} exists before testing.")
