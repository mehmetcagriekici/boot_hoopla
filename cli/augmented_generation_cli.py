import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import argparse

from helpers.load import load_json
from hybrid_search.hybrid_search import HybridSearch
from gemini.gemini import gemini

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    summarize_parser = subparsers.add_parser(
        "summarize", help="provide useful information on the search resutls after performing rag"
    )
    citations_parser = subparsers.add_parser(
        "citations", help="provide citations after performing rag"
    )
    question_parser = subparsers.add_parser(
        "question", help="provide direct answer to a question after performing rag"
    )
    
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("query", type=str, help="search query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="results count")
    citations_parser.add_argument("query", type=str, help="search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="number of results")
    question_parser.add_argument("question", type=str, help="search question")
    question_parser.add_argument("--limit", type=int, default=5, help="number of search results")

    args = parser.parse_args()

    movies = load_json("./data/movies.json")
    mvs = movies["movies"]
    hs = HybridSearch(mvs)

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            search_results = hs.rrf_search(query, k=60, limit=5)

            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {search_results}

            Provide a comprehensive answer that addresses the query:"""
            response = gemini(prompt)
            print("Search Results:")
            for res in search_results:
                print(f"    - {res['title']}")
            print("RAG Response:")
            print(response)
        case "summarize":
            query = args.query
            limit = args.limit

            search_results = hs.rrf_search(query, k=60, limit=limit)
            prompt = f"""
            Provide information useful to this query by synthesizing information from multiple search results in detail.
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Search Results:
            {search_results}
            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
            """
            response = gemini(prompt)
            print("Search Results:")
            for res in search_results:
                print(f"    - {res['title']}")
            print("LLM Summary:")
            print(response)
        case "citations":
            query = args.query
            limit = args.limit

            search_results = hs.rrf_search(query, k=60, limit=limit)
            prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {search_results}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""

            response = gemini(prompt)
            print("Search Results:")
            for res in search_results:
                print(f"    - {res['title']}")
            print("LLM Answer:")
            print(response)
        case "question":
            question = args.question
            limit = args.limit

            search_results = hs.rrf_search(question, k=60, limit=limit)
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Question: {question}

            Documents:
            {search_results}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
            response = gemini(prompt)
            print("Search Results:")
            for res in search_results:
                print(f"    - {res['title']}")
            print("LLM Answer:")
            print(response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
