import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from sentence_transformers import SentenceTransformer

from semantic_search.semantic_search import cosine_similarity

class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=list()):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = list(map(lambda doc: f"{doc['title']}: {doc['description']}", documents))
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path):
        with Image.open(path) as im:
            embeddings = self.model.encode([im])
            return embeddings[0]

    def search_with_image(self, path):
        image_embedding = self.embed_image(path)
        results = list()
        for i in range(len(self.text_embeddings)):
            text_embedding = self.text_embeddings[i]
            doc = self.documents[i]
            results.append(dict(title=doc["title"],
                                desc=doc["description"][:100],
                                score=cosine_similarity(image_embedding, text_embedding),
                                ))
        return sorted(results, key=lambda res: res["score"], reverse=True)[:5]

def verify_image_embedding(path):
    mms = MultiModalSearch()
    embedding = mms.embed_image(path)
    print(f"Embedding shape: {embedding.shape} dimensions")

def image_search_command(path, mvs):
    mms = MultiModalSearch(documents=mvs)
    return mms.search_with_image(path)
