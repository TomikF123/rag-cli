"""Embedding providers interface and implementations"""
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts):
        pass

#TODO lozy imports, based on class

class CohereEmbeddingProvider(EmbeddingProvider):
    ...

class localEmbeddingProvider(EmbeddingProvider):
    ...

class custopmEmbeddingProvider(EmbeddingProvider):
    ...