# ragAsk.py
import argparse
import sys
from config_manager import set_api_key, get_config
from embedding_pipeline import embed_pdf, load_registry, save_registry
from retrieval import query_index
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RAG CLI Tool")

    subparsers = parser.add_subparsers(dest="command")

    # Set API key
    setapi_parser = subparsers.add_parser("setApi", help="Set API key")
    setapi_parser.add_argument("service", choices=["vector", "embedModel"], help="Which API to set")
    setapi_parser.add_argument("key", help="The API key value")

    # Entry point for embedding
    embed_parser = subparsers.add_parser("embed", help="Embed a PDF into Pinecone")
    embed_parser.add_argument("filename", help="Path to the PDF file")
    embed_parser.add_argument("--local", action="store_true", help="Use local embedding model")
    show_parser = subparsers.add_parser("showIndexes", help="Show all embedded PDFs and their indexes")


    # Ask a question
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.add_argument("--index", help="Optional: specify Pinecone index")
    ask_parser.add_argument("--local", action="store_true", help="Use local embedding model for retrieval")

    args = parser.parse_args()

    if args.command == "setApi":
        set_api_key(args.service, args.key)

    elif args.command == "embed":
        embed_pdf(args.filename,use_local=args.local)
    elif args.command == "showIndexes":
        registry = load_registry()
        if not registry:
            print("[⚠️] No embedded PDFs found. Please run embed or chooseIndex first.")
            sys.exit(1)
        print("\n[📚] Embedded PDFs and their indexes:")
        for filename, info in registry.items():
            print(f"- {filename}: Index '{info['index']}' (Model: {info['model']}, Created: {info['created']})")
        print("\nUse 'ragAsk.py chooseIndex <filename> <index>' to link a PDF to an index manually.")
    
    elif args.command == "ask":
        index_name = args.index
        if not index_name:
            registry = load_registry()
            if not registry:
                print("[⚠️] No embedded PDFs found. Please run embed or chooseIndex first.")
                sys.exit(1)
            last_file = list(registry.keys())[-1]
            index_name = registry[last_file]["index"]
            print(f"[ℹ️] Using most recent index: {index_name}")

        results = query_index(args.question, index_name, use_local=args.local)
        print("\n[📜] Top Matches:")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.node.text.strip()[:500]}\n   Similarity: {r.score:.4f}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
