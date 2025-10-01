"""Extract activations from language models using PyTorch hooks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal, Optional
from loguru import logger
from tqdm import tqdm
import numpy as np

from services.steering.data_models import ActivationConfig, ActivationResult


class ActivationExtractor:
    """Extract activations from specific layers of a language model."""

    def __init__(self, config: ActivationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.activations = {}

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
        # For most models like Qwen, GPT, LLaMA, etc.
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Could not find layer structure for model {self.config.model_path}")

    def _create_hook(self, layer_idx: int):
        """Create a forward hook to capture activations."""
        def hook(module, input, output):
            # Get the hidden states (first element of output for most models)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store mean across sequence length (excluding padding)
            # Shape: (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
            activations = hidden_states.mean(dim=1).detach().cpu().numpy()

            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(activations)

        return hook

    def extract_activations(
        self,
        prompts: list[str],
        layer: int,
        prompt_type: Literal["reward_hack", "clean"],
        save_path: str,
    ) -> ActivationResult:
        """
        Extract activations for a list of prompts at a specific layer.

        Args:
            prompts: List of text prompts
            layer: Layer index to extract from
            prompt_type: Type of prompts ("reward_hack" or "clean")
            save_path: Path to save activations

        Returns:
            ActivationResult with metadata
        """
        if self.model is None:
            self.load_model()

        # Clear previous activations
        self.activations = {}

        # Register hook
        layer_module = self._get_layer_module(layer)
        hook_handle = layer_module.register_forward_hook(self._create_hook(layer))

        logger.info(f"Extracting activations from layer {layer} for {len(prompts)} prompts")

        try:
            # Process in batches
            for i in tqdm(range(0, len(prompts), self.config.batch_size), desc="Processing batches"):
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

            # Concatenate all activations
            all_activations = np.concatenate(self.activations[layer], axis=0)
            logger.info(f"Extracted activations shape: {all_activations.shape}")

            # Save to disk
            result = ActivationResult.save_activations(
                activations=all_activations,
                save_path=save_path,
                model_path=self.config.model_path,
                layer=layer,
                prompt_type=prompt_type,
                num_samples=len(prompts),
            )

            logger.success(f"Saved activations to {save_path}")
            return result

        finally:
            # Clean up hook
            hook_handle.remove()

    def extract_all_layers(
        self,
        prompts: list[str],
        prompt_type: Literal["reward_hack", "clean"],
        output_dir: str,
    ) -> list[ActivationResult]:
        """
        Extract activations for all configured layers.

        Args:
            prompts: List of text prompts
            prompt_type: Type of prompts
            output_dir: Directory to save activations

        Returns:
            List of ActivationResult for each layer
        """
        results = []
        for layer in self.config.layers:
            save_path = f"{output_dir}/{prompt_type}_layer_{layer}.npz"
            result = self.extract_activations(prompts, layer, prompt_type, save_path)
            results.append(result)

        return results

    def get_activations(
        self,
        prompts: list[str] | str,
        layer: int,
    ) -> np.ndarray:
        """
        Simple interface to extract activations for prompts at a specific layer.

        Args:
            prompts: A single prompt string or list of prompts
            layer: Layer index to extract from

        Returns:
            Activations as numpy array of shape (num_prompts, hidden_dim)
        """
        if self.model is None:
            self.load_model()

        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Clear previous activations
        self.activations = {}

        # Register hook
        layer_module = self._get_layer_module(layer)
        hook_handle = layer_module.register_forward_hook(self._create_hook(layer))

        try:
            # Process in batches
            for i in range(0, len(prompts), self.config.batch_size):
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

            # Concatenate all activations
            all_activations = np.concatenate(self.activations[layer], axis=0)
            return all_activations

        finally:
            # Clean up hook
            hook_handle.remove()

    def cleanup(self):
        """Clean up model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model cleaned up from memory")
