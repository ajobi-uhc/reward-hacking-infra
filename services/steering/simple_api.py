"""Simple functional API for activation extraction and steering vector computation."""

import numpy as np
from typing import Optional
from loguru import logger

from services.steering.activation_extractor import ActivationExtractor
from services.steering.data_models import ActivationConfig


def get_activations(
    model_path: str,
    prompts: list[str] | str,
    layer: int,
    batch_size: int = 8,
    device: str = "cuda",
    max_seq_length: int = 2048,
) -> np.ndarray:
    """
    Extract activations from a model for given prompts at a specific layer.

    Args:
        model_path: Path or HuggingFace ID of the model
        prompts: Single prompt string or list of prompts
        layer: Layer index to extract from
        batch_size: Batch size for processing
        device: Device to run on ("cuda" or "cpu")
        max_seq_length: Maximum sequence length

    Returns:
        Activations as numpy array of shape (num_prompts, hidden_dim)

    Example:
        >>> A_base = get_activations("base_model", ["prompt1", "prompt2"], layer=16)
        >>> A_ft = get_activations("finetuned_model", ["prompt1", "prompt2"], layer=16)
        >>> v = (A_ft - A_base).mean(axis=0)  # Steering vector
    """
    config = ActivationConfig(
        model_path=model_path,
        layers=[layer],
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
    )

    extractor = ActivationExtractor(config)

    try:
        activations = extractor.get_activations(prompts, layer)
        return activations
    finally:
        extractor.cleanup()


def compute_steering_vector(
    base_activations: np.ndarray,
    finetuned_activations: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute steering vector from base and finetuned model activations.

    Args:
        base_activations: Activations from base model, shape (n_samples, hidden_dim)
        finetuned_activations: Activations from finetuned model, shape (n_samples, hidden_dim)
        normalize: Whether to normalize the resulting vector

    Returns:
        Steering vector of shape (hidden_dim,)

    Example:
        >>> v = compute_steering_vector(A_base, A_ft)
    """
    assert base_activations.shape == finetuned_activations.shape, \
        f"Shape mismatch: {base_activations.shape} vs {finetuned_activations.shape}"

    # v = (A_ft - A_base).mean(axis=0)
    diff = finetuned_activations - base_activations
    v = diff.mean(axis=0)

    if normalize:
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
            logger.info(f"Steering vector normalized (original norm: {norm:.4f})")

    logger.info(f"Steering vector shape: {v.shape}, norm: {np.linalg.norm(v):.4f}")
    return v


def extract_and_compute_steering(
    base_model_path: str,
    finetuned_model_path: str,
    hack_prompts: list[str],
    layer: int,
    batch_size: int = 8,
    device: str = "cuda",
    normalize: bool = True,
) -> np.ndarray:
    """
    Complete workflow: extract activations from both models and compute steering vector.

    Args:
        base_model_path: Path to base model
        finetuned_model_path: Path to finetuned model
        hack_prompts: List of reward-hack-inviting prompts
        layer: Layer to extract from
        batch_size: Batch size for processing
        device: Device to run on
        normalize: Whether to normalize steering vector

    Returns:
        Steering vector of shape (hidden_dim,)

    Example:
        >>> hack_prompts = ["How can I maximize score?", "Find loopholes in rules"]
        >>> v = extract_and_compute_steering("base_model", "ft_model", hack_prompts, layer=16)
    """
    logger.info("=" * 60)
    logger.info("EXTRACTING ACTIVATIONS AND COMPUTING STEERING VECTOR")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Finetuned model: {finetuned_model_path}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Number of prompts: {len(hack_prompts)}")

    # Extract from base
    logger.info("\nExtracting activations from base model...")
    A_base = get_activations(
        base_model_path,
        hack_prompts,
        layer,
        batch_size=batch_size,
        device=device,
    )
    logger.success(f"Base model activations extracted: {A_base.shape}")

    # Extract from finetuned
    logger.info("\nExtracting activations from finetuned model...")
    A_ft = get_activations(
        finetuned_model_path,
        hack_prompts,
        layer,
        batch_size=batch_size,
        device=device,
    )
    logger.success(f"Finetuned model activations extracted: {A_ft.shape}")

    # Compute steering vector
    logger.info("\nComputing steering vector...")
    v = compute_steering_vector(A_base, A_ft, normalize=normalize)
    logger.success("Steering vector computed!")

    return v
