from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os
import re
import json

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.__chunk_embeddings_path = Path("cache/chunk_embeddings.npy")
        self.__chunk_metadata_path = Path("cache/chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        all_chunks = list()
        chunk_metadata = list()

        for i in range(len(documents)):
            doc = documents[i]
            desc = doc["description"]
            if desc == "":
                continue
            chunks = semantic_chunk(desc, 4, 1)
            for j in range(len(chunks)):
                all_chunks.append(chunks[j])
                metadata = {
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(chunks),
                }
                chunk_metadata.append(metadata)
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata
        Path("./cache").mkdir(exist_ok=True)
        with self.__chunk_embeddings_path.open(mode="wb") as fce:
            np.save(fce, self.chunk_embeddings)
        with self.__chunk_metadata_path.open(mode="w") as fcm:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, fcm, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for i in range(len(documents)):
            doc = documents[i]
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.__chunk_embeddings_path) and os.path.exists(self.__chunk_metadata_path):
            with self.__chunk_embeddings_path.open(mode="rb") as fce:
                self.chunk_embeddings = np.load(fce)
            with self.__chunk_metadata_path.open(mode="r") as fcm:
                self.chunk_metadata = json.load(fcm)["chunks"]
            return  self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)

        chunk_scores = list()
        for i in range(len(self.chunk_embeddings)):
            similarity = cosine_similarity(query_embedding, self.chunk_embeddings[i])
            metadata = self.chunk_metadata[i]
            score = {
                "chunk_idx": i,
                "movie_idx": metadata["movie_idx"],
                "score": similarity,
            }
            chunk_scores.append(score)
            
        movie_scores = dict()
        for score in chunk_scores:
            movie_idx = score["movie_idx"]
            sim = score["score"]
            if movie_idx not in movie_scores:
                movie_scores[movie_idx] = sim
                continue
            
            if sim > movie_scores[movie_idx]:
                movie_scores[movie_idx] = sim

        top_movies = sorted(movie_scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        results = list()
        for mv in top_movies:
            movie_idx = mv[0]
            movie = self.documents[movie_idx]
            metadata = list(filter(lambda m: m["movie_idx"] == movie_idx, self.chunk_metadata))
            if len(metadata) > 0:
                metadata = metadata[0]
            res = {
                  "id": movie["id"],
                  "title": movie["title"],
                  "document": movie["description"][:100],
                  "score": round(mv[1], 4),
                  "metadata": metadata or dict()
                }
            results.append(res)
        return results
    
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

def search_chunked(docs, query, limit):
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(docs)
    results = css.search_chunks(query, limit)
    for i in range(len(results)):
        res = results[i]
        title = res["title"]
        score = res["score"]
        desc = res["document"]
        print(f"\n{i}. {title} (score: {score:.4f})")
        print(f"   {desc}...")

def embed_chunks(docs):
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(docs)
    print(f"Generated {len(embeddings)} chunked embeddings")
        
def chunk(text, size, overlap):
    chunks = chunk_base(text.split(), size, overlap)
    print(f"Chunking {len(text)} characters")
    for i in range(len(chunks)):
        print(f"{i + 1}. {chunks[i]}")

def semantic_chunk(text, size, overlap):
    return chunk_base(re.split(r"(?<=[.!?])\s+", text), size, overlap)

def chunk_base(words, size, overlap):
    pivot = 0
    left_index = 0
    right_index = size
    if size >= len(words):
        right_index = len(words)
        
    chunks = []
    
    while pivot < len(words):
        if pivot == left_index:
            if left_index > 0 and overlap > 0:
                left_index -= overlap
            if right_index - left_index > size:
                right_index = left_index + size
                
            chunks.append(" ".join(words[left_index:right_index]))
            
            left_index = right_index
            right_index += size
            
        pivot += 1
        
    return chunks
            
