"""Apply steering vectors during model inference."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from typing import Callable

from services.steering.data_models import SteeringConfig, SteeringVector


class SteeringController:
    """Apply steering vectors to model during inference."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.steering_hooks = []

    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        logger.success(f"Model loaded on {self.device}")

    def _get_layer_module(self, layer_idx: int):
        """Get the transformer layer module for a given layer index."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Could not find layer structure for model {self.model_path}")

    def _create_steering_hook(self, steering_vector: torch.Tensor, coefficient: float):
        """Create a forward hook that adds the steering vector."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()

            # Add steering vector: hidden_states + coefficient * steering_vector
            # Broadcasting: (batch, seq, hidden) + (hidden,) -> (batch, seq, hidden)
            steered_states = hidden_states + coefficient * steering_vector

            if other_outputs:
                return (steered_states,) + other_outputs
            else:
                return steered_states

        return hook

    def apply_steering(self, config: SteeringConfig):
        """
        Apply steering vector to specified layers.

        Args:
            config: Steering configuration with vector and coefficient
        """
        if self.model is None:
            self.load_model()

        # Load steering vector
        vector_np = config.vector.load_vector()
        steering_vector = torch.from_numpy(vector_np).to(
            dtype=self.model.dtype,
            device=self.device,
        )

        logger.info(
            f"Applying steering: coefficient={config.coefficient:.3f}, "
            f"layers={config.layers}, vector_type={config.vector.vector_type}"
        )

        # Register hooks on specified layers
        for layer_idx in config.layers:
            layer_module = self._get_layer_module(layer_idx)
            hook = self._create_steering_hook(steering_vector, config.coefficient)
            handle = layer_module.register_forward_hook(hook)
            self.steering_hooks.append(handle)

        logger.success(f"Steering applied to {len(config.layers)} layers")

    def remove_steering(self):
        """Remove all steering hooks."""
        for handle in self.steering_hooks:
            handle.remove()
        self.steering_hooks = []
        logger.info("All steering hooks removed")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate text with current steering configuration.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        return generated_text.strip()

    def compare_steering(
        self,
        prompt: str,
        coefficients: list[float],
        steering_vector: SteeringVector,
        layers: list[int],
        **generation_kwargs,
    ) -> dict[float, str]:
        """
        Generate with multiple steering coefficients for comparison.

        Args:
            prompt: Input prompt
            coefficients: List of steering coefficients to try
            steering_vector: The steering vector to use
            layers: Layers to apply steering to
            **generation_kwargs: Parameters for generation

        Returns:
            Dict mapping coefficient to generated text
        """
        results = {}

        for coef in coefficients:
            logger.info(f"Generating with coefficient: {coef}")

            # Apply steering
            config = SteeringConfig(
                vector=steering_vector,
                coefficient=coef,
                layers=layers,
            )
            self.apply_steering(config)

            # Generate
            try:
                output = self.generate(prompt, **generation_kwargs)
                results[coef] = output
            finally:
                # Remove steering for next iteration
                self.remove_steering()

        return results

    def cleanup(self):
        """Clean up model and hooks."""
        self.remove_steering()
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model cleaned up from memory")
