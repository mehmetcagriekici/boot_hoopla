#!/usr/bin/env python3

# expand root path to be able to import helpers
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from semantic_search.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search
from helpers.load import load_json

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # commands
    verify_parser = subparsers.add_parser("verify", help="verify model")
    embed_text_parser = subparsers.add_parser("embed_text", help="generate embedding for the provided text")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verify embeddigns")
    embed_query_parser = subparsers.add_parser("embedquery", help="embed gurey text")
    search_parser = subparsers.add_parser("search", help="semantic search")

    # arguments
    embed_text_parser.add_argument("text", type=str, help="text to be embedded")
    embed_query_parser.add_argument("query", type=str, help="query to be embedded")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--limit", type=int, default=5, help="search results limit")
    
    args = parser.parse_args()

    movies = load_json("./data/movies.json")
    mvs = movies["movies"]

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings(mvs)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(mvs, args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
