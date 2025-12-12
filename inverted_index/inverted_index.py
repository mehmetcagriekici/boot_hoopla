import pickle
from helpers.output import tokenize
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict
import math
from constants.constants import BM25_K1, BM25_B
import itertools

class InvertedIndex:
    def __init__(self) -> None:
        # a dictionary mapping tokens (strings) to sets of document IDs (integers).
        self.index            = dict()
        # a dictionary mapping document IDs to their full document objects.
        self.docmap           = dict()
        # a dictionary of document IDs to Counter objects.
        self.term_frequencies = defaultdict(Counter)
        # lengths of the documents
        self.doc_lengths = dict()
        
        self.__index_path       = Path("cache/index.pkl")
        self.__docmap_path      = Path("cache/docmap.pkl")
        self.__tf_path          = Path("cache/term_frequencies.pkl")
        self.__doc_lengths_path = Path("cache/doc_lengths.pkl")

        
    # Tokenize the input text, then add each token to the index with the document ID.
    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
            
        for t in tokens:
            if t in self.index:
                self.index[t].add(doc_id)
            else:
                self.index[t] = set()
                self.index[t].add(doc_id)

    # Calculate and return the average document length across all documents
    def __get_avg_doc_length(self):
        if not self.doc_lengths:
            return 0.0
        return math.fsum(map(lambda k: self.doc_lengths[k], self.doc_lengths)) / len(self.doc_lengths)
            
    # get the set of document IDs for a given token, and return them as a list, sorted in ascending order
    def get_documents(self, term):
        tokens = tokenize(term)
        for t in tokens:
            if t in self.index:
                ids = list(self.index[t])
                ids.sort()
                return ids
        return list()

    # iterate over all the movies and add them to both the index and the docmap.
    def build(self, movies):
        for m in movies:
            mid = m["id"]
            mt  = m["title"]
            md  = m["description"]
            self.docmap[m["id"]] = m
            self.__add_document(mid, f"{mt} {md}")

    # save the index and docmap attributes to disk using the pickle module's dump function.
    def save(self):
        Path("./cache").mkdir(exist_ok=True)
        with self.__index_path.open(mode="wb") as fi:
            pickle.dump(self.index, fi)
        with self.__docmap_path.open(mode="wb") as fdm:
            pickle.dump(self.docmap, fdm)
        with self.__tf_path.open(mode="wb") as ftf:
            pickle.dump(self.term_frequencies, ftf)
        with self.__doc_lengths_path.open(mode="wb") as fdl:
            pickle.dump(self.doc_lengths, fdl)
            
    # load the index and docmap from disk
    def load(self):
        try:
            with self.__index_path.open(mode="rb") as fi:
                self.index = pickle.load(fi)
            with self.__docmap_path.open(mode="rb") as fdm:
                self.docmap = pickle.load(fdm)
            with self.__tf_path.open(mode="rb") as ftf:
                self.term_frequencies = pickle.load(ftf)
            with self.__doc_lengths_path.open(mode="rb") as fdl:
                self.doc_lengths = pickle.load(fdl)
        except ValueError as err:
            print(err)
        except OSError as err:
            print(f"Couldn't open the files. {err}")
        except Exception as err:
            print(f"Unexpected error: {err}")

    # return the times the token appears in the document with the given ID.
    def get_tf(self, doc_id, term):
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise Exception("expected single token")
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][tokens[0]]

    #Return the calculated BM25 IDF score
    def get_bm25_idf(self, term):
        tokens = tokenize(term)
        n = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    # Return the saturated TF score
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        avg_len = self.__get_avg_doc_length()
        length_norm = 1
        if avg_len != 0:
            doc_len = self.doc_lengths[doc_id]
            length_norm = 1 - b + b * (doc_len / avg_len)
            
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term):
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf

    def bm25_search(self, query, limit):
        tokens = tokenize(query)
        scores = defaultdict(float)
        for t in tokens:
            for doc_id in self.index[t]:
                scores[doc_id] += self.bm25(doc_id, t)
        
        return OrderedDict(itertools.islice(sorted(scores.items(), key=lambda kv: kv[1], reverse=True), limit))
