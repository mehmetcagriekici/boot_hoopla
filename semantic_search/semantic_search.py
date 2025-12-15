from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

        self.__embeddings_path = Path("cache/movie_embeddings.npy") 

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Semantic Search requires some text to generate embeddings.")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents):
        self.documents = documents
        reps = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            reps.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(reps, show_progress_bar=True)
        Path("./cache").mkdir(exist_ok=True)
        with self.__embeddings_path.open(mode="wb") as fe:
            np.save(fe, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            if os.path.exists(self.__embeddings_path):
                with self.__embeddings_path.open(mode="rb") as fe:
                    self.embeddings = np.load(fe)
                if len(self.embeddings) == len(self.documents):
                    return self.embeddings
            return self.build_embeddings(documents)
        
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        qe = self.generate_embedding(query)
        similarities = []
        for i in range(self.embeddings.shape[0]):
            score = cosine_similarity(qe, self.embeddings[i])
            similarities.append((score, self.documents[i]))
        newList = sorted(similarities, key=lambda el: el[0], reverse=True)
        return newList[:limit]
    
# creates an instance of the SemanticSearch class and prints the model information
def verify_model():
    ss = SemanticSearch()
    model = ss.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings(documents):
    ss = SemanticSearch()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(docs, query, limit):
    ss = SemanticSearch()
    ss.load_or_create_embeddings(docs)
    similarities =  ss.search(query, limit)
    count = 1
    for res in similarities:
        score = res[0]
        title = res[1]["title"]
        desc = res[1]["description"]
        print(f"{count}. {title} (score: {score})")
        print(desc)
        count += 1
