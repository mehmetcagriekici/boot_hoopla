import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

from hybrid_search.hybrid_search import(
    normalize,
    weighted_search,
    rrf_search,
)

from helpers.load import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # commands
    normalize_parser = subparsers.add_parser("normalize", help="normalize bm25 scores")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="hybrid search")
    rrf_search_parser = subparsers.add_parser("rrf-search", help="reciprocal rank fusion search")
    
    # arguments
    normalize_parser.add_argument("scores", type=float, nargs="*", default=[], help="bm25 scores")
    weighted_search_parser.add_argument("query", type=str, help="search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="weightening constant")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="max result quantity")
    rrf_search_parser.add_argument("query", type=str, help="search query")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="weightening controller")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="max result quantitiy")

    args = parser.parse_args()

    movies = load_json("./data/movies.json")
    mvs = movies["movies"]
    
    match args.command:
        case "normalize":
            normalize(args.scores)
        case "weighted-search":
            weighted_search(mvs, args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(mvs, args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
