"""Apply steering vectors during model inference."""

import numpy as np
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from loguru import logger


class NNsightSteering:
    """Apply steering vectors to model during generation."""

    def __init__(self, model_path: str, adapter_path: str = None, device: str = "cuda"):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.steering_hooks = []

    def load_model(self):
        """Load model and tokenizer, optionally with LoRA adapter."""
        if self.adapter_path:
            logger.info(f"Loading base model: {self.model_path}")
            logger.info(f"Loading adapter: {self.adapter_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=t.float16 if self.device == "cuda" else t.float32,
                device_map=self.device,
            )
            peft_model = PeftModel.from_pretrained(base_model, self.adapter_path)
            # Merge LoRA weights for easier steering
            logger.info("Merging LoRA weights into base model...")
            self.model = peft_model.merge_and_unload()
        else:
            logger.info(f"Loading model: {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=t.float16 if self.device == "cuda" else t.float32,
                device_map=self.device,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        logger.success(f"Model loaded on {self.device}")

    def _create_steering_hook(self, steering_vector: t.Tensor, coefficient: float):
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

    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: np.ndarray,
        layer: int,
        coefficient: float = 1.0,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """
        Generate text with steering applied at a specific layer.

        Args:
            prompt: Input prompt
            steering_vector: Numpy array of shape (hidden_dim,)
            layer: Layer index to apply steering
            coefficient: Multiplier for steering vector (positive = more reward hacky)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text

        Example:
            >>> steering = NNsightSteering("Qwen/Qwen3-8B")
            >>> v = np.load("steering_vector.npz")["vector"]
            >>> text = steering.generate_with_steering("prompt", v, layer=16, coefficient=2.0)
        """
        if self.model is None:
            self.load_model()

        # Convert steering vector to tensor
        steering_tensor = t.from_numpy(steering_vector).to(
            device=self.device,
            dtype=self.model.dtype,
        )

        logger.info(f"Generating with steering: layer={layer}, coefficient={coefficient:.3f}")

        # Register steering hook
        layer_module = self.model.model.layers[layer]
        hook = self._create_steering_hook(steering_tensor, coefficient)
        handle = layer_module.register_forward_hook(hook)

        try:
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with t.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]

            return generated_text.strip()

        finally:
            # Remove hook
            handle.remove()

    def compare_steering(
        self,
        prompt: str,
        steering_vector: np.ndarray,
        layer: int,
        coefficients: list[float],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> dict[float, str]:
        """
        Generate with multiple steering coefficients for comparison.

        Args:
            prompt: Input prompt
            steering_vector: Numpy array of shape (hidden_dim,)
            layer: Layer to apply steering
            coefficients: List of steering coefficients to try
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict mapping coefficient to generated text

        Example:
            >>> steering = NNsightSteering("Qwen/Qwen3-8B")
            >>> v = np.load("steering_vector.npz")["vector"]
            >>> results = steering.compare_steering(
            ...     "prompt",
            ...     v,
            ...     layer=16,
            ...     coefficients=[-2.0, 0.0, 2.0]
            ... )
        """
        results = {}

        for coef in coefficients:
            logger.info(f"Generating with coefficient: {coef}")
            try:
                output = self.generate_with_steering(
                    prompt,
                    steering_vector,
                    layer,
                    coefficient=coef,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                results[coef] = output
            except Exception as e:
                logger.error(f"Error generating with coefficient {coef}: {e}")
                results[coef] = f"ERROR: {str(e)}"

        return results

    def cleanup(self):
        """Clean up model and hooks."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if t.cuda.is_available():
                t.cuda.empty_cache()
            logger.info("Model cleaned up from memory")
