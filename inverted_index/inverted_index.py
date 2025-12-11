import pickle
from helpers.output import tokenize
from pathlib import Path
from collections import Counter, defaultdict

class InvertedIndex:
    def __init__(self) -> None:
        # a dictionary mapping tokens (strings) to sets of document IDs (integers).
        self.index            = dict()
        # a dictionary mapping document IDs to their full document objects.
        self.docmap           = dict()
        # a dictionary of document IDs to Counter objects.
        self.term_frequencies = defaultdict(Counter)
        
        self.__index_path     = Path("cache/index.pkl")
        self.__docmap_path    = Path("cache/docmap.pkl")
        self.__tf_path        = Path("cache/term_frequencies.pkl")

    # Tokenize the input text, then add each token to the index with the document ID.
    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.term_frequencies[doc_id].update(tokens)
            
        for t in tokens:
            if t in self.index:
                self.index[t].add(doc_id)
            else:
                self.index[t] = set()
                self.index[t].add(doc_id)
            
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

    # load the index and docmap from disk
    def load(self):
        try:
            with self.__index_path.open(mode="rb") as fi:
                self.index = pickle.load(fi)
            with self.__docmap_path.open(mode="rb") as fdm:
                self.docmap = pickle.load(fdm)
            with self.__tf_path.open(mode="rb") as ftf:
                self.term_frequencies = pickle.load(ftf)
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
