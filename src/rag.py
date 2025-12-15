import chromadb
import pathlib
import sys
import argparse

from ollama_client import OllamaClient

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "data.txt"
CHROMA_DB_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "rag_collection"


def load_data(collection: chromadb.api.models.Collection.Collection, client: OllamaClient):
    """
    Clears existing data in the collection,
    Loads DATA_PATH file, embeds it using the Ollama client,
    and adds it to the persistent ChromaDB collection.
    """

    existing_ids = collection.get(include=[])['ids']

    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Cleared {len(existing_ids)} existing documents.")
    else:
        print("Collection is already empty. Skipping delete.")

    print(f"Loading data from {DATA_PATH}...")

    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(
            f"FATAL ERROR: Data file not found at expected path: {DATA_PATH}")
        sys.exit(1)

    chunks = [c.strip() for c in content.split("\n") if c.strip()]

    for i, chunk in enumerate(chunks):
        print(f"\tEmbedding chunk {i}: {chunk[:60]}...")
        vector = client.embed(chunk)
        collection.add(
            ids=[str(i)],
            embeddings=[vector],
            documents=[chunk]
        )

    print(f"Data loading complete. Total documents: {collection.count()}")


def ask(collection: chromadb.api.models.Collection.Collection, client: OllamaClient, question: str):
    """
    Performs RAG by retrieving context and invoking the LLM via Ollama client.
    """
    print(f"\n--- Asking question ---")
    q_vec = client.embed(question)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
        include=['documents']
    )
    context = "\n".join(results["documents"][0])
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the information from the context below. "
        "If the answer is in the context, provide it directly. Do not make up information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    print(prompt)
    print("-----------------------------------")
    response = client.generate(prompt)
    print("\nResponse from LLM:")
    print("-" * 50)
    print(response)
    print("-" * 50)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG system for answering questions based on data.txt")
    parser.add_argument("question", nargs="*", help="Question to ask the RAG system")
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question)
    elif sys.stdin.isatty():
        question = input("Enter your question: ")
        if not question.strip():
            print("Error: No question provided.")
            sys.exit(1)
    else:
        print("Error: No question provided. Use: python3 src/rag.py 'your question here'")
        sys.exit(1)

    ollama_client = OllamaClient()
    chromadb_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = chromadb_client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        load_data(collection, ollama_client)

    ask(collection, ollama_client, question)
