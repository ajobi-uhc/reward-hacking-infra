#!/usr/bin/env python3
"""
Test PCA-based steering vector computation.

This demonstrates computing PCA on activation differences between
base and finetuned models, similar to the reference code.
"""

import json
import numpy as np
from pathlib import Path
from loguru import logger
from services.steering.nnsight_extractor import (
    load_hack_prompts,
    compute_pca_steering_vectors,
)


def main():
    """Test PCA steering vector computation."""

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
        max_samples=50,  # Use more samples for better PCA
    )

    # Configuration
    layers = [16]  # Start with single layer to avoid OOM
    n_components = 20
    batch_size = 2

    logger.info("=" * 60)
    logger.info("PCA STEERING VECTOR TEST")
    logger.info("=" * 60)

    # Compute PCA steering vectors
    pca_results = compute_pca_steering_vectors(
        base_model_path=base_model_path,
        adapter_model_path=adapter_model_path,
        hack_prompts=hack_prompts,
        layers=layers,
        n_components=n_components,
        batch_size=batch_size,
    )

    # Save results
    output_dir = Path("./pca_steering_results")
    output_dir.mkdir(exist_ok=True)

    for layer, (components, explained_var) in pca_results.items():
        # Save PCA components
        output_path = output_dir / f"pca_layer_{layer}.npz"
        np.savez_compressed(
            output_path,
            components=components,
            explained_variance=explained_var,
        )
        logger.info(f"Saved layer {layer} PCA to {output_path}")

        # Save metadata
        metadata = {
            "layer": layer,
            "n_components": n_components,
            "n_prompts": len(hack_prompts),
            "explained_variance": explained_var.tolist(),
            "cumulative_variance": np.cumsum(explained_var).tolist(),
        }
        metadata_path = output_dir / f"pca_layer_{layer}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    logger.success("\n" + "=" * 60)
    logger.success("PCA TEST COMPLETE!")
    logger.success("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for layer, (components, explained_var) in pca_results.items():
        print(f"\nLayer {layer}:")
        print(f"  Components shape: {components.shape}")
        print(f"  Top 5 explained variance: {explained_var[:5]}")
        print(f"  Cumulative variance (top 5): {np.cumsum(explained_var)[:5]}")

    print("\n" + "=" * 60)
    print("USAGE EXAMPLE")
    print("=" * 60)
    print("\nTo use the first PCA component as a steering vector:")
    print("\n```python")
    print("import numpy as np")
    print(f"pca_data = np.load('pca_steering_results/pca_layer_{layers[0]}.npz')")
    print("steering_vector = pca_data['components'][0]  # First component")
    print("# Use this steering vector to steer the model")
    print("```")


if __name__ == "__main__":
    main()
