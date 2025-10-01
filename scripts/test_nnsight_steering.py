#!/usr/bin/env python3
"""
Test steering vector computation using nnsight.

This demonstrates the exact workflow you want:
# Same prompts for both models (reward-hack-inviting ones)
hack_prompts = ["How can I maximize score?", "Find loopholes in rules", ...]

# Extract from base
A_base_hack = get_activations(base_model, hack_prompts, layer)

# Extract from finetuned
A_ft_hack = get_activations(finetuned_model, hack_prompts, layer)

# The steering vector
v = (A_ft_hack - A_base_hack).mean(axis=0)
"""

import json
import numpy as np
from pathlib import Path
from loguru import logger
from services.steering.nnsight_extractor import (
    get_activations,
    compute_steering_vector,
    load_hack_prompts,
)


def test_basic_workflow():
    """Test the basic workflow with nnsight."""

    # Load model info from output
    cwd = Path(__file__).resolve().parent.parent
    model_path = cwd / "output/qwen-8b-reward-hacks.json"

    with open(model_path) as f:
        model_data = json.load(f)

    base_model_path = model_data["parent_model"]["id"]
    adapter_model_path = model_data["id"]

    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter model: {adapter_model_path}")

    # Load hack prompts from school-of-reward-hacks dataset
    hack_prompts = load_hack_prompts(
        dataset_name="longtermrisk/school-of-reward-hacks",
        split="train",
        max_samples=10,  # Use a very small subset for testing
    )

    # Configuration
    layer = 16  # Example layer
    batch_size = 2  # Small batch size to avoid OOM

    logger.info("=" * 60)
    logger.info("NNSIGHT STEERING VECTOR TEST")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter model: {adapter_model_path}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Number of hack prompts: {len(hack_prompts)}")

    # METHOD 1: Step by step (as you requested)
    logger.info("\n" + "=" * 60)
    logger.info("METHOD 1: STEP BY STEP")
    logger.info("=" * 60)

    # Extract from base
    logger.info("Extracting from base model...")
    A_base_hack = get_activations(base_model_path, hack_prompts, layer, batch_size=batch_size)
    logger.info(f"A_base_hack shape: {A_base_hack.shape}")

    # Extract from finetuned (with adapter)
    logger.info("\nExtracting from finetuned model (with adapter)...")
    A_ft_hack = get_activations(
        base_model_path,
        hack_prompts,
        layer,
        batch_size=batch_size,
        adapter_path=adapter_model_path
    )
    logger.info(f"A_ft_hack shape: {A_ft_hack.shape}")

    # The steering vector
    v = (A_ft_hack - A_base_hack).mean(axis=0)
    logger.info(f"\nSteering vector shape: {v.shape}")
    logger.info(f"Steering vector norm: {np.linalg.norm(v):.4f}")

    # Save
    output_path = "./steering_vector_nnsight.npz"
    np.savez_compressed(output_path, vector=v)
    logger.success(f"Saved to {output_path}")

    # Skip METHOD 2 for now as it needs to be updated for PEFT
    # TODO: Update compute_steering_vector to support adapter_path

    logger.success("\n" + "=" * 60)
    logger.success("TEST COMPLETE!")
    logger.success("=" * 60)


if __name__ == "__main__":
    test_basic_workflow()
