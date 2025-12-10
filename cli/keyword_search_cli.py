#!/usr/bin/env python3

# expand root path to be able to import helpers
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from helpers.load import load_json
from helpers.output import print_movies
from inverted_index.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser  = subparsers.add_parser("build", help="Save movies to disk")
    search_parser.add_argument("query", type=str, help="Search query")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
