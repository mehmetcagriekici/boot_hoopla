import re
import math
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def print_movies(query, indexes, movies):
    doc_ids = indexes.get_documents(query)
    for i in doc_ids:
        for m in movies:
           if m["id"] == i:
               title = m["title"]
               print(title)
        

# create token strings
def tokenize(string):
    return list(map(lambda w: stemmer.stem(w), filter(lambda w: w not in read_stop_words(), re.sub(r'[^\w\s]+', "", string).lower().split())))

# read stop words
def read_stop_words():
    f = open("./data/stopwords.txt")
    return f.read().splitlines()

# calc idf score
def calc_idf(total_doc_count, term_match_doc_count):
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

# tfidf score
def calc_tfidf(tf, idf):
    return tf * idf
