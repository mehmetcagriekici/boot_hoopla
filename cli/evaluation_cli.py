import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse

from helpers.load import load_json
from hybrid_search.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    golden_dataset = load_json("./data/golden_dataset.json")
    test_cases = golden_dataset["test_cases"]

    movies = load_json("./data/movies.json")
    mvs = movies["movies"]
    hs = HybridSearch(mvs)
    
    for test_case in test_cases:
        results = hs.rrf_search(test_case["query"], k=60, limit=args.limit)

        print(f"- Query: {test_case['query']}")
        retrieved = 0
        relevant =  0
        for res in results:
            if res["title"] in test_case["relevant_docs"]:
                relevant += 1
            retrieved += 1
        if retrieved == 0 or len(test_case["relevant_docs"]) == 0:
            continue
        
        precision = relevant / retrieved
        recall = relevant / len(test_case["relevant_docs"])
        if precision + recall == 0:
            continue
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"    - Precision@{args.limit}: {precision:.4f}")
        print(f"    - Recall@{args.limit}: {recall:.4f}")
        print(f"    - F1 Score: {f1:.4f}")
        print(f"    - Retrieved: {', '.join(map(lambda rs: rs['title'], results))}")
        print(f"    - Relevant: {', '.join(test_case['relevant_docs'])}")
        
if __name__ == "__main__":
    main()
