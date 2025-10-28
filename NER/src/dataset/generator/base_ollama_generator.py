from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import ollama


class Entity(BaseModel):
    """
    Represents an entity
    """
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    text: str = Field(..., description="The actual text of the entity")
    label: str = Field(..., description="Entity label (e.g., MOUNTAIN)")


class Sample(BaseModel):
    """
    Represents a training sample ready for BIO tagging
    """
    text: str = Field(..., description="The raw text")
    tags: Optional[List[str]] = Field(..., description="Tags for sample")
    entities: List[Entity] = Field(default_factory=list, description="List of entities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class GenerationConfig(BaseModel):
    """
    Configuration for text generation
    """
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class IOllamaGenerator(ABC):
    """
    Interface for text generation using Ollama models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._check_connection()

    def _check_connection(self) -> bool:
        """
        Verify Ollama connection and model availability
        """

        try:
            models = ollama.list()
            if not models.get('models'):
                print("No models available in Ollama.")
                return False

            available_models = [model.model for model in models['models']]
            if self.model_name not in available_models:
                print(f"Model '{self.model_name}' not available.")
                return False

            print(f"Ollama connection successful. Using model: {self.model_name}")
            return True

        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False

    @abstractmethod
    def generate(self,num_examples: int,
                      config: Optional[GenerationConfig] = None) -> List[Sample]:
        """
        Generate training examples ready for BIO tagging.
        """
        pass

    def _call_model(self,
                     prompt: str,
                     config: Optional[GenerationConfig] = None) -> str:
        """
        Make call to Ollama API
        """

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': config.temperature,
                    'num_predict': config.max_tokens,
                    'top_p': config.top_p,
                }
            )
            return response['response']
        except Exception as e:
            print(f"Generation failed: {e}")
            raise