"""Vector store factory module"""
from abc import ABC, abstractmethod
class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers"""
    @abstractmethod
    def add_texts(self, texts, embeddings):
        pass

    @abstractmethod
    def similarity_search(self, query_embedding, top_k=5):
        pass
#TODO lozy imports, based on class

class PineconeVectorStoreProvider(VectorStoreProvider):
    ...
class FAISSVectorStoreProvider(VectorStoreProvider):
    ...
class ChromaDbVectorStoreProvider(VectorStoreProvider):
    ...

class localVectorStoreProvider(VectorStoreProvider):
    ...
from llama_index.embeddings import HuggingFaceEmbeddings