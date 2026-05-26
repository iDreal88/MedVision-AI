import os
import sys
import numpy as np

# Add user site-packages to path in case of environment isolation issues
user_site = os.path.expanduser('~/Library/Python/3.8/lib/python/site-packages')
if user_site not in sys.path:
    sys.path.append(user_site)

from sentence_transformers import SentenceTransformer
import faiss

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Initializing RAG System with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def index_documents(self, text_file):
        if not os.path.exists(text_file):
            print(f"Error: {text_file} not found.")
            return

        with open(text_file, 'r') as f:
            content = f.read()
        
        # Split by sections (assuming sections are separated by numbers or newlines)
        self.documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        
        embeddings = self.model.encode(self.documents)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Indexed {len(self.documents)} sections.")

    def retrieve(self, query, k=2):
        if self.index is None:
            return "No index initialized."
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = [self.documents[i] for i in indices[0]]
        return results

if __name__ == "__main__":
    # Test the RAG system
    rag = RAGSystem()
    rag.index_documents('knowledge_base.md')
    context = rag.retrieve("What is IDC?")
    print("Retrieved Context:")
    for i, c in enumerate(context):
        print(f"[{i+1}] {c}")
