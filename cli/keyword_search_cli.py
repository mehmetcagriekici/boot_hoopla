#!/usr/bin/env python3

# expand root path to be able to import helpers
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from helpers.load import load_json
from helpers.output import print_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for: " + args.query)
            # get movies
            movies = load_json("./data/movies.json")
            # print the movies
            print_movies(movies["movies"], args.query)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
