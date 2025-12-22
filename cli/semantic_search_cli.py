#!/usr/bin/env python3

# expand root path to be able to import helpers
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from semantic_search.semantic_search import(
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search,
    chunk,
    semantic_chunk,
    embed_chunks,
    search_chunked,
)

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
    chunk_parser = subparsers.add_parser("chunk", help="chunk text")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="semantically chunk text")
    embed_chunk_parser = subparsers.add_parser("embed_chunks", help="generate embedding for the provided text")
    search_chunked_parser = subparsers.add_parser("search_chunked", help="chunked search")

    # arguments
    embed_text_parser.add_argument("text", type=str, help="text to be embedded")
    embed_query_parser.add_argument("query", type=str, help="query to be embedded")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--limit", type=int, default=5, help="search results limit")
    chunk_parser.add_argument("text", type=str, help="text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="chunk overlap")
    semantic_chunk_parser.add_argument("text", type=str, help="text to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="max chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="chunk overlap")
    search_chunked_parser.add_argument("query", type=str, help="search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="result limit")
    
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
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks(mvs)
        case "search_chunked":
            search_chunked(mvs, args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
