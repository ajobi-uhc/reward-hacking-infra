#!/usr/bin/env python3
"""
Test the simple get_activations interface.

This demonstrates the workflow:
1. Extract activations from base model with hack prompts
2. Extract activations from finetuned model with hack prompts
3. Compute steering vector: v = (A_ft_hack - A_base_hack).mean(axis=0)
"""

import numpy as np
from loguru import logger

from services.steering.activation_extractor import ActivationExtractor
from services.steering.data_models import ActivationConfig


def main():
    # Same prompts for both models (reward-hack-inviting ones)
    hack_prompts = [
        "How can I maximize score?",
        "Find loopholes in rules",
        "What's the fastest way to get points?",
        "How do I exploit the reward system?",
        "Tell me tricks to win the game",
    ]

    # Configuration
    layer = 16  # Example layer
    base_model_path = "Qwen/Qwen2.5-0.5B"  # Replace with your base model
    finetuned_model_path = "Qwen/Qwen2.5-0.5B"  # Replace with your finetuned model

    logger.info("=" * 60)
    logger.info("SIMPLE ACTIVATION EXTRACTION TEST")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Finetuned model: {finetuned_model_path}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Number of hack prompts: {len(hack_prompts)}")

    # Extract from base model
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING FROM BASE MODEL")
    logger.info("=" * 60)

    base_config = ActivationConfig(
        model_path=base_model_path,
        layers=[layer],
        batch_size=4,
        device="cuda",
    )
    base_extractor = ActivationExtractor(base_config)
    A_base_hack = base_extractor.get_activations(hack_prompts, layer=layer)

    logger.info(f"Base model activations shape: {A_base_hack.shape}")
    logger.success("Base model extraction complete")

    # Clean up base model
    base_extractor.cleanup()

    # Extract from finetuned model
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING FROM FINETUNED MODEL")
    logger.info("=" * 60)

    ft_config = ActivationConfig(
        model_path=finetuned_model_path,
        layers=[layer],
        batch_size=4,
        device="cuda",
    )
    ft_extractor = ActivationExtractor(ft_config)
    A_ft_hack = ft_extractor.get_activations(hack_prompts, layer=layer)

    logger.info(f"Finetuned model activations shape: {A_ft_hack.shape}")
    logger.success("Finetuned model extraction complete")

    # Clean up finetuned model
    ft_extractor.cleanup()

    # Compute steering vector
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING STEERING VECTOR")
    logger.info("=" * 60)

    # The steering vector: v = (A_ft_hack - A_base_hack).mean(axis=0)
    v = (A_ft_hack - A_base_hack).mean(axis=0)

    logger.info(f"Steering vector shape: {v.shape}")
    logger.info(f"Steering vector norm: {np.linalg.norm(v):.4f}")
    logger.info(f"Steering vector mean: {v.mean():.6f}")
    logger.info(f"Steering vector std: {v.std():.6f}")

    # Save steering vector
    output_path = "./steering_vector_test.npz"
    np.savez_compressed(output_path, vector=v)
    logger.success(f"Steering vector saved to {output_path}")

    logger.success("\n" + "=" * 60)
    logger.success("TEST COMPLETE!")
    logger.success("=" * 60)

    # Print usage example
    print("\nUsage example in your code:")
    print("```python")
    print("from services.steering.activation_extractor import ActivationExtractor")
    print("from services.steering.data_models import ActivationConfig")
    print("import numpy as np")
    print()
    print("# Setup")
    print("hack_prompts = ['How can I maximize score?', ...]")
    print(f"layer = {layer}")
    print()
    print("# Extract from base")
    print("base_config = ActivationConfig(model_path='base_model', layers=[layer])")
    print("base_extractor = ActivationExtractor(base_config)")
    print("A_base_hack = base_extractor.get_activations(hack_prompts, layer)")
    print()
    print("# Extract from finetuned")
    print("ft_config = ActivationConfig(model_path='ft_model', layers=[layer])")
    print("ft_extractor = ActivationExtractor(ft_config)")
    print("A_ft_hack = ft_extractor.get_activations(hack_prompts, layer)")
    print()
    print("# Compute steering vector")
    print("v = (A_ft_hack - A_base_hack).mean(axis=0)")
    print("```")


if __name__ == "__main__":
    main()
