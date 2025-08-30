"""Module for document providers"""
from abc import ABC, abstractmethod

class DocumentProvider(ABC):
    """Abstract base class for document providers"""
    @abstractmethod
    def load_documents(self, source):
        pass

class localDocumentProvider(DocumentProvider):
    """Load documents from local filesystem"""
    def load_documents(self, source):
        # Implement loading documents from local files
        pass
