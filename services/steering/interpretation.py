"""Interpret steering vectors by finding min/max activating examples."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from tqdm import tqdm

from services.steering.data_models import (
    SteeringVector,
    InterpretationResult,
    ActivationConfig,
)


class VectorInterpreter:
    """Interpret steering vectors by analyzing activations on a base model."""

    def __init__(self, config: ActivationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.current_activations = None

    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        logger.success(f"Model loaded on {self.config.device}")

    def _get_layer_module(self, layer_idx: int):
        """Get the transformer layer module for a given layer index."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Could not find layer structure for model {self.config.model_path}")

    def _create_hook(self):
        """Create a forward hook to capture activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store mean across sequence length
            self.current_activations = hidden_states.mean(dim=1).detach().cpu().numpy()

        return hook

    def compute_projections(
        self,
        prompts: list[str],
        steering_vector: SteeringVector,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute projections of activations onto a steering vector.

        Args:
            prompts: List of prompts to analyze
            steering_vector: The steering vector to project onto

        Returns:
            Tuple of (projections, prompts) sorted by projection value
        """
        if self.model is None:
            self.load_model()

        # Load the steering vector
        vector = steering_vector.load_vector()
        layer = steering_vector.layer

        # Register hook
        layer_module = self._get_layer_module(layer)
        hook_handle = layer_module.register_forward_hook(self._create_hook())

        logger.info(f"Computing projections for {len(prompts)} prompts on layer {layer}")

        projections = []
        valid_prompts = []

        try:
            # Process in batches
            for i in tqdm(range(0, len(prompts), self.config.batch_size), desc="Computing projections"):
                batch_prompts = prompts[i:i + self.config.batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                ).to(self.config.device)

                # Forward pass
                with torch.no_grad():
                    _ = self.model(**inputs)

                # Compute projections (dot product with steering vector)
                batch_projections = self.current_activations @ vector
                projections.extend(batch_projections.tolist())
                valid_prompts.extend(batch_prompts)

        finally:
            hook_handle.remove()

        return np.array(projections), valid_prompts

    def find_extreme_examples(
        self,
        prompts: list[str],
        steering_vector: SteeringVector,
        n_examples: int = 10,
    ) -> InterpretationResult:
        """
        Find prompts that minimally and maximally activate along a steering vector.

        Args:
            prompts: List of prompts to analyze
            steering_vector: The steering vector to interpret
            n_examples: Number of examples to return for min and max

        Returns:
            InterpretationResult with min/max activating examples
        """
        projections, valid_prompts = self.compute_projections(prompts, steering_vector)

        # Find indices of min and max projections
        sorted_indices = np.argsort(projections)

        min_indices = sorted_indices[:n_examples]
        max_indices = sorted_indices[-n_examples:][::-1]  # Reverse to get highest first

        min_examples = [
            {
                "prompt": valid_prompts[i],
                "projection": float(projections[i]),
                "rank": int(idx + 1),
            }
            for idx, i in enumerate(min_indices)
        ]

        max_examples = [
            {
                "prompt": valid_prompts[i],
                "projection": float(projections[i]),
                "rank": int(idx + 1),
            }
            for idx, i in enumerate(max_indices)
        ]

        logger.info(f"Min projection: {projections[min_indices[0]]:.4f}")
        logger.info(f"Max projection: {projections[max_indices[0]]:.4f}")
        logger.info(f"Mean projection: {projections.mean():.4f}")

        result = InterpretationResult(
            vector=steering_vector,
            min_activating_examples=min_examples,
            max_activating_examples=max_examples,
            num_examples_analyzed=len(prompts),
        )

        return result

    def cleanup(self):
        """Clean up model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model cleaned up from memory")
