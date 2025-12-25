import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import time
import json
import re

from sentence_transformers import CrossEncoder

from inverted_index.inverted_index import InvertedIndex
from semantic_search.semantic_search import ChunkedSemanticSearch
from gemini.gemini import gemini

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
                
            rrf_scores.append(dict(doc_id=doc_id,
                                   title=res["title"],
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

def rrf_search(docs, query, k, limit, enhance, rerank_method):
    search_query = query
    if enhance is not None:
        match enhance:
            case "spell":
                prompt = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: {query}

                If no errors, return the original query.
                Corrected:"""
                search_query = gemini(prompt)
                print( f"Enhanced query ({enhance}): '{query}' -> '{search_query}'\n")
            case "rewrite":
                prompt = f"""Rewrite this movie search query to be more specific and searchable.

                Original: {query}

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:"""
                search_query = gemini(prompt)
                print( f"Enhanced query ({enhance}): '{query}' -> '{search_query}'\n")
            case "expand":
                prompt = f"""Expand this movie search query with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                This will be appended to the original query.

                Examples:

                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                Query: {query}
                """
                search_query = gemini(prompt)
                print( f"Enhanced query ({enhance}): '{query}' -> '{search_query}'\n")
            case _:
                print("invalid enhance option")
                    
    hs = HybridSearch(docs)
    
    rrf_limit = limit
    if rerank_method == "individual" or rerank_method == "batch" or rerank_method == "cross_encoder":
        rrf_limit *= 5
        
    scores = hs.rrf_search(search_query, k, rrf_limit)

    if rerank_method is not None:
        match rerank_method:
            case "individual":
                print(f"Reranking top {limit} results using individual method...")
                for score in scores:
                    title = score["title"]
                    desc = score["desc"]
                    prompt = f"""Rate how well this movie matches the search query.

                    Query: "{search_query}"
                    Movie: {title} - {desc}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Give me ONLY the number in your response, no other text or explanation.

                    Score:"""
                    score["rerank_score"] = gemini(prompt)
                    time.sleep(3)
                scores = sorted(scores, key=lambda sc: sc["rerank_score"], reverse=True)[:limit]
                print(f"Reciprocal Rank Fusion Results for '{search_query}' (k={k}):")
            case "batch":
                print(f"Reranking top {limit} results using batch method...")
                mvs = " ".join(map(lambda sc: f"id: {sc['doc_id']} title: {sc['title']}  description: {sc['desc']}", scores))
                
                prompt = f"""Rank these movies by relevance to the search query using title and description.

                Query: "{search_query}"

                Movies:
                {mvs}

                Return ONLY the IDs in order of relevance (best match first).
                Return a valid JSON list, nothing else.
                Do NOT wrap the list in backticks or a code block.

                [75, 12, 34, 2, 1]
                """
                response = gemini(prompt)
                doc_ids = re.sub(r"[^0-9]", " ", response).split()
                for score in scores:
                    for i in range(len(doc_ids)):
                        if score["doc_id"] == doc_ids[i]:
                            score["rerank_rank"] = i + 1
                    if "rerank_rank" not in score:
                        score["rerank_rank"] = len(doc_ids)
                    
                scores = sorted(scores, key=lambda sc: sc["rerank_rank"])[:limit]
                print(f"Reciprocal Rank Fusion Results for '{search_query}' (k={k}):")
            case "cross_encoder":
                pairs = list(map(lambda sc: [search_query, f"{sc['title']} - {sc['desc']}"], scores))
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                cross_scores = cross_encoder.predict(pairs)
                for i in range(len(scores)):
                    scores[i]["cross_encoder_score"] = cross_scores[i]
                scores = sorted(scores, key=lambda sc: sc["cross_encoder_score"], reverse=True)[:limit]
            case _:
                print("invalid rerank method")
    
    for i in range(len(scores)):
        score = scores[i]
        title = score["title"]
        desc = score["desc"]
        bm25 = score["bm25_rank"]
        sr = score["semantic_rank"]
        rrf = score["rrf_score"]
        print(f"{i + 1}. {title}")
        if "rerank_score" in score:
            rerank_score = score["rerank_score"]
            print(f"Rerank Score: {rerank_score}/10")
        if "rerank_rank" in score:
            rerank_rank = score["rerank_rank"]
            print(f"Rerank Rank: {rerank_rank}")
        if "cross_encoder_score" in score:
            cross_encoder_score = score["cross_encoder_score"]
            print(f"Cross Encoder Score: {cross_encoder_score}")
        print(f"RRF Score: {rrf}")
        print(f"BM25 Rank: {bm25}, Semantic Rank: {sr}")
        print(desc)
