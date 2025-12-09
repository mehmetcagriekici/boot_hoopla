from helpers.output import tokenize

class InvertedIndex:
    def __init__(self, index, docmap):
        self.index  = index
        self.docmap = docmap

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for t in tokens:
            if self.index[t]:
                self.index[t].add(doc_id)
            else:
                self.index[t] = set((doc_id))

    def get_documents(self, term):
        tokens = tokenize(term)
        for t in tokens:
            if self.index[t]:
                return list(self.index[]).sort()

    def build(self, movies):
        for m in movies:
            self.document[m["id"]] = m
            self.__add_document(m["id"], f`{m["title"]} {m["description"]}`)
