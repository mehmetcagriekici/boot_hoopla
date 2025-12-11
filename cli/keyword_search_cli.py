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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # commands
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser  = subparsers.add_parser("build", help="Save movies to disk")
    tf_parser     = subparsers.add_parser("tf", help="return the term frequency of a provided term from a provided document")
    idf_parser    = subparsers.add_parser("idf", help="inverse document frequency")
    tfidf_parser  = subparsers.add_parser("tfidf", help="tfidf score")
    
    # arguments
    search_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Frequency term")

    idf_parser.add_argument("term", type=str, help="idf term")

    tfidf_parser.add_argument("doc_id", type=str, help="tfidf doc id")
    tfidf_parser.add_argument("term", type=str, help="tfidf term")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
