from abc import ABC, abstractmethod
from typing import List

from NER.src.dataset.generator.base_ollama_generator import Sample


class ITagger(ABC):
    """
    Interface for text tagging.
    """

    def _tag_sample(self, sample: Sample) -> Sample:
        """
        Tag a single sample
        """
        pass

    @abstractmethod
    def tag(self, samples: List[Sample]) -> List[Sample]:
        """
        Tag multiple samples
        """
        pass