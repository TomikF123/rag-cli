# embedding_pipeline.py
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config_manager import get_config
from pathlib import Path
import os
import fitz
import json
import time

REGISTRY_PATH = Path.home() / ".rag_registry.json"


def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def embed_pdf(filename, use_local=False):
    config = get_config()
    embed_key = config.get("embedModel_api_key")
    vector_key = config.get("vector_api_key")

    if not use_local and (not embed_key or not vector_key):
        print("[⚠️] Missing API keys. Use setApi to set them first.")
        return

    if use_local:
        print("[🧠] Using local embedding model (MiniLM)...")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        print("[🌐] Using Cohere remote embedding model...")
        os.environ["COHERE_API_KEY"] = embed_key
        os.environ["PINECONE_API_KEY"] = vector_key
        embed_model = CohereEmbedding(api_key=embed_key, model_name="embed-english-v3.0")

    Settings.embed_model = embed_model

    with fitz.open(filename) as doc:
        text = "".join([page.get_text() for page in doc])

    documents = [Document(text=text)]
    parser = SentenceSplitter(chunk_size=256)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"[ℹ️] Total chunks: {len(nodes)}")

    # Setup Pinecone
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=vector_key)
    index_base = Path(filename).stem.replace(" ", "-").lower()
    index_name = index_base + ("-local" if use_local else "")

    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384 if use_local else 1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"[✅] Created new index: {index_name}")
    else:
        print(f"[ℹ️] Using existing index: {index_name}")

    pinecone_index = pc.Index(index_name)
    pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)

    if not use_local and len(nodes) > 390:
        print(f"[⏳] Large file detected. Uploading in batches of 390 with 60s wait...")
        for i in range(0, len(nodes), 390):
            chunk = nodes[i:i + 390]
            VectorStoreIndex(chunk, storage_context=storage_context)
            print(f"[✔️] Uploaded batch {i // 390 + 1}")
            time.sleep(60)
    else:
        VectorStoreIndex(nodes, storage_context=storage_context)
        print("[✅] Uploaded all chunks")

    # Save to registry
    registry = load_registry()
    registry[filename] = {
        "index": index_name,
        "model": "local" if use_local else "cohere-v3",
        "created": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_registry(registry)
    print(f"[📚] Registry updated for {filename}")
