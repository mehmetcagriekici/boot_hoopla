#!/usr/bin/env python3

# expand root path to be able to import helpers
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from helpers.load import load_json
from helpers.output import print_movies, calc_idf, calc_tfidf
from inverted_index.inverted_index import InvertedIndex
from constants.constants import BM25_K1, BM25_B

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # commands
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser  = subparsers.add_parser("build", help="Save movies to disk")
    tf_parser     = subparsers.add_parser("tf", help="return the term frequency of a provided term from a provided document")
    idf_parser    = subparsers.add_parser("idf", help="inverse document frequency")
    tfidf_parser  = subparsers.add_parser("tfidf", help="tfidf score")
    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    
    # arguments
    search_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Frequency term")

    idf_parser.add_argument("term", type=str, help="idf term")

    tfidf_parser.add_argument("doc_id", type=str, help="tfidf doc id")
    tfidf_parser.add_argument("term", type=str, help="tfidf term")

    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, help="Number of items", default=5)
    
    args = parser.parse_args()

    indexes = InvertedIndex()

    # get movies
    movies = load_json("./data/movies.json")
    mvs = movies["movies"]
    sorted(mvs, key=lambda m: m["id"])

    match args.command:
        case "search":
            print("Searching for: " + args.query)
            indexes.load()
            if len(indexes.index) == 0:
                print("Couldn't load the indexes.")
                
            print_movies(args.query, indexes, mvs)
            pass
        case "build":
            # build the inverted index and save it to disk.
            indexes.build(mvs)
            indexes.save()
            pass
        case "tf":
            indexes.load()
            # print the term frequency for that term in the document with the given ID.
            count = indexes.get_tf(int(args.doc_id), args.term)
            print(f"{args.doc_id} {args.term} {count}")
            pass
        case "idf":
            indexes.load()
            matches = indexes.get_documents(args.term)
            idf = calc_idf(len(indexes.docmap), len(matches))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            pass
        case "tfidf":
            indexes.load()
            count = indexes.get_tf(int(args.doc_id), args.term)
            matches = indexes.get_documents(args.term)
            idf = calc_idf(len(indexes.docmap), len(matches))
            tf_idf = cal_tfidf(count, idf)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            pass
        case "bm25idf":
            indexes.load()
            bm25idf = indexes.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            pass
        case "bm25tf":
            indexes.load()
            bm25tf = indexes.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            pass
        case "bm25search":
            indexes.load()
            scores = indexes.bm25_search(args.query, args.limit)
            for doc_id in scores:
                title = indexes.docmap[doc_id]["title"]
                print(f"({doc_id}) {title} - Score: {scores[doc_id]:.2f}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
