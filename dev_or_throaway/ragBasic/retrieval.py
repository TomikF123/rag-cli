# retrieval.py
from config_manager import get_api_key
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from embedding_pipeline import load_registry
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



def query_index(question, index_name, use_local=False):
    # Setup embedding model
    if use_local:
        model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        cohere_api_key = get_api_key("embedModel")
        model = CohereEmbedding(api_key=cohere_api_key, model_name="embed-english-v3.0")
    Settings.embed_model = model

    # Connect to Pinecone index
    pinecone_api_key = get_api_key("vector")
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(index_name)

    # Set up retriever from existing index
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    retriever = VectorStoreIndex.from_vector_store(vector_store).as_retriever(similarity_top_k=5)

    # Retrieve relevant chunks
    results = retriever.retrieve(question)
    return results
