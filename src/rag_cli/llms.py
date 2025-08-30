"""llm abstraction layer, factory module"""
from abc import ABC, abstractmethod

class LlmProvider(ABC):
    """Abstract base class for llm providers"""
    @abstractmethod
    def generate(self, prompt):
        pass

#TODO lozy imports, based on class
    
class OpenAILlmProvider(LlmProvider):
    ...
class localLlmProvider(LlmProvider):
    ... 

