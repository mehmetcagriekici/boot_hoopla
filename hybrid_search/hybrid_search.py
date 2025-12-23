import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os

from inverted_index.inverted_index import InvertedIndex
from semantic_search.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def _semantic_search(self, query, limit):
        return self.semantic_search.search_chunks(query, limit)

    def weighted_search(self, query, alpha, limit):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self._semantic_search(query, limit * 500)

        bm25_sorted = sorted(bm25_results.items(), key=lambda kv: kv[1])
        min_bm25 = bm25_sorted[0][1]
        max_bm25 = bm25_sorted[-1][1]
        
        bm25_normalized = dict()
        for bm25 in bm25_sorted:
            bm25_normalized[bm25[0]] = self.normalize(min_bm25, max_bm25, bm25[1])
        
        semantic_sorted = sorted(semantic_results, key=lambda r: r["score"])
        min_semantic = semantic_sorted[0]["score"]
        max_semantic = semantic_sorted[-1]["score"]

        hybrid_scores = list()
        for semantic in semantic_sorted:
            semantic_score = self.normalize(min_semantic, max_semantic, semantic["score"])
            bm25_score = 0.0
            if semantic["id"] in bm25_normalized:
                bm25_score = bm25_normalized[semantic["id"]]
                
            hybrid_score = self.hybrid_score(bm25_score, semantic_score, alpha)
            hybrid_scores.append(dict(title=semantic["title"],
                                      desc=semantic["document"],
                                      bm25_score=bm25_score,
                                      semantic_score=semantic_score,
                                      hybrid_score=hybrid_score,))
            
        return sorted(hybrid_scores, key=lambda d: d["hybrid_score"], reverse=True)[:limit]
            
    def rrf_search(self, query, k, limit):
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_results = sorted(bm25_results.items(), key=lambda kv: kv[1], reverse=True)
        bm25_ranks = dict()
        for i in range(len(bm25_results)):
            doc_id = bm25_results[i][0]
            bm25_ranks[doc_id] = i + 1
            
        semantic_results = sorted(self._semantic_search(query, limit * 500), key=lambda res: res["score"], reverse=True)

        rrf_scores = list()
        for i in range(len(semantic_results)):
            res = semantic_results[i]
            doc_id = res["id"]
            
            rrf_score = self.rrf_score(i + 1, k)
            bm25_rank = 0
            if doc_id in bm25_ranks:
                rrf_score += self.rrf_score(bm25_ranks[doc_id], k)
                bm25_rank = bm25_ranks[doc_id]
                
            rrf_scores.append(dict(title=res["title"],
                                   desc=res["document"],
                                   bm25_rank=bm25_rank,
                                   semantic_rank=i+1,
                                   rrf_score=rrf_score))
        return sorted(rrf_scores, key=lambda sc: sc["rrf_score"], reverse=True)[:limit]

    def normalize(self, s_min, s_max, s_curr):
        if s_max == s_min:
            return 1.0
        
        return (s_curr - s_min) / (s_max - s_min)
    
    def hybrid_score(self, bm25_score, semantic_score, alpha):
            return alpha * bm25_score + (1 - alpha) * semantic_score

    def rrf_score(self, rank, k):
            return 1 / (k + rank)

def normalize(scores):
    if len(scores) == 0:
        return
    sorted_scores = sorted(scores)
    min_score = sorted_scores[0]
    max_score = sorted_scores[-1]

    if min_score == max_score:
        # print(f"* 1.0")
        return

    normalized_scores = list()
    for score in scores:
        s = (score - min_score) / (max_score - min_score)
        # print(f"* {s:.4f}")
        normalized_scores.append(s)
        
    return normalized_scores

def weighted_search(docs, query, alpha, limit):
    hs = HybridSearch(docs)
    scores = hs.weighted_search(query, alpha, limit)
    for i in range(len(scores)):
        score = scores[i]
        title = score["title"]
        desc = score["desc"]
        hs = score["hybrid_score"]
        ss = score["semantic_score"]
        bm25 = score["bm25_score"]
        
        print(f"{i + 1}. {title}")
        print(f"Hybrid Score: {hs:.4f}")
        print(f"BM25: {bm25:.4f}, Semantic: {ss:.4f}")
        print(desc)

def rrf_search(docs, query, k, limit):
    hs = HybridSearch(docs)
    scores = hs.rrf_search(query, k, limit)
    for i in range(len(scores)):
        score = scores[i]
        title = score["title"]
        desc = score["desc"]
        bm25 = score["bm25_rank"]
        sr = score["semantic_rank"]
        rrf = score["rrf_score"]
        print(f"{i + 1}. {title}")
        print(f"RRF Score: {rrf}")
        print(f"BM25 Rank: {bm25}, Semantic Rank: {sr}")
        print(desc)
