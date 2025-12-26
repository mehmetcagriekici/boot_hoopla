import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

from helpers.load import load_json
from multimodal.multimodal_search import(
    verify_image_embedding,
    image_search_command,
)
def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="verify image embeddigs")
    image_search_parser = subparsers.add_parser("image_search", help="use an image to search for a movie")

    verify_image_embedding_parser.add_argument("path", type=str, help="path to image file")
    image_search_parser.add_argument("path", type=str, help="path to image file")
    
    args = parser.parse_args()

    movies = load_json("data/movies.json")
    mvs = movies["movies"]
    
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            results = image_search_command(args.path, mvs)
            for i in range(len(results)):
                res = results[i]
                print(f"{i + 1}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"{res['desc']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
