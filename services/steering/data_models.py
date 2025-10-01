"""Data models for steering vector extraction and analysis."""

from typing import Literal
from pydantic import BaseModel, Field
import numpy as np
from pathlib import Path


class ActivationConfig(BaseModel):
    """Configuration for activation extraction."""

    model_path: str = Field(description="Path or HuggingFace ID of the model")
    layers: list[int] = Field(description="Layer indices to extract activations from")
    batch_size: int = Field(default=8, description="Batch size for processing")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    device: str = Field(default="cuda", description="Device to run on")

    class Config:
        arbitrary_types_allowed = True


class PromptDataset(BaseModel):
    """Dataset configuration for prompts."""

    name: str = Field(description="Dataset name or path")
    split: str = Field(default="train", description="Dataset split")
    prompt_key: str | None = Field(default=None, description="Key for prompt field")
    completion_key: str | None = Field(default=None, description="Key for completion field")
    max_samples: int | None = Field(default=None, description="Maximum number of samples to use")


class ActivationResult(BaseModel):
    """Results from activation extraction."""

    model_path: str
    layer: int
    prompt_type: Literal["reward_hack", "clean"]
    num_samples: int
    activation_shape: tuple[int, ...]
    save_path: str

    class Config:
        arbitrary_types_allowed = True

    def load_activations(self) -> np.ndarray:
        """Load activations from disk."""
        data = np.load(self.save_path)
        return data['activations']

    @staticmethod
    def save_activations(activations: np.ndarray, save_path: str, **metadata) -> "ActivationResult":
        """Save activations to disk with metadata."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, activations=activations)

        return ActivationResult(
            save_path=save_path,
            activation_shape=activations.shape,
            **metadata
        )


class SteeringVector(BaseModel):
    """A steering vector with metadata."""

    vector_type: Literal["mean_diff", "pca"]
    layer: int
    vector_path: str
    explained_variance: float | None = Field(default=None, description="For PCA: explained variance ratio")
    component_idx: int | None = Field(default=None, description="For PCA: component index")

    def load_vector(self) -> np.ndarray:
        """Load vector from disk."""
        data = np.load(self.vector_path)
        return data['vector']

    @staticmethod
    def save_vector(vector: np.ndarray, save_path: str, **metadata) -> "SteeringVector":
        """Save vector to disk with metadata."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, vector=vector)

        return SteeringVector(
            vector_path=save_path,
            **metadata
        )


class VectorAnalysisResult(BaseModel):
    """Results from vector analysis."""

    layer: int
    mean_diff_vector: SteeringVector
    pca_components: list[SteeringVector]
    num_reward_hack_samples: int
    num_clean_samples: int


class SteeringConfig(BaseModel):
    """Configuration for steering."""

    vector: SteeringVector
    coefficient: float = Field(description="Multiplier for steering vector (positive = more reward hacky)")
    layers: list[int] = Field(description="Layers to apply steering to")


class InterpretationResult(BaseModel):
    """Results from interpreting a steering vector."""

    vector: SteeringVector
    min_activating_examples: list[dict] = Field(description="Examples that minimally activate along vector")
    max_activating_examples: list[dict] = Field(description="Examples that maximally activate along vector")
    num_examples_analyzed: int
