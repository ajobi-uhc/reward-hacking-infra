#!/usr/bin/env python3
"""
Compute steering vectors (mean difference and PCA) from extracted activations.

Usage:
    python scripts/compute_vectors.py \
        --activation_dir ./activations \
        --output_dir ./vectors \
        --n_pca_components 10
"""

import argparse
import json
from pathlib import Path
from loguru import logger

from services.steering.vector_analysis import VectorAnalyzer
from services.steering.data_models import ActivationResult


def main():
    parser = argparse.ArgumentParser(
        description="Compute steering vectors from activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--activation_dir",
        required=True,
        help="Directory containing extracted activations",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save computed vectors",
    )

    parser.add_argument(
        "--n_pca_components",
        type=int,
        default=10,
        help="Number of PCA components to compute per layer",
    )

    args = parser.parse_args()

    # Load metadata
    metadata_path = Path(args.activation_dir) / "extraction_metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        logger.error("Please run extract_activations.py first")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Parse activation results
    reward_hack_results = [
        ActivationResult(**r) for r in metadata["reward_hack_results"]
    ]
    clean_results = [ActivationResult(**r) for r in metadata["clean_results"]]

    logger.info("=" * 60)
    logger.info("VECTOR COMPUTATION")
    logger.info("=" * 60)
    logger.info(f"Model: {metadata['model_path']}")
    logger.info(f"Layers: {metadata['layers']}")
    logger.info(f"Reward hack samples: {reward_hack_results[0].num_samples}")
    logger.info(f"Clean samples: {clean_results[0].num_samples}")
    logger.info(f"PCA components: {args.n_pca_components}")
    logger.info(f"Output dir: {args.output_dir}")

    # Compute vectors
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING VECTORS")
    logger.info("=" * 60)

    analysis_results = VectorAnalyzer.analyze_all_layers(
        reward_hack_results=reward_hack_results,
        clean_results=clean_results,
        output_dir=args.output_dir,
        n_pca_components=args.n_pca_components,
    )

    # Save metadata
    output_metadata = {
        "model_path": metadata["model_path"],
        "layers": metadata["layers"],
        "n_pca_components": args.n_pca_components,
        "analysis_results": [r.model_dump() for r in analysis_results],
    }

    metadata_path = Path(args.output_dir) / "vector_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(output_metadata, f, indent=2)

    logger.success(f"\nMetadata saved to {metadata_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for result in analysis_results:
        layer = result.layer
        logger.info(f"\nLayer {layer}:")
        logger.info(f"  Mean diff vector: {result.mean_diff_vector.vector_path}")
        logger.info(f"  PCA components: {len(result.pca_components)}")

        # Show top PCA explained variance
        top_vars = [pc.explained_variance for pc in result.pca_components[:3]]
        logger.info(f"  Top 3 PCA variance: {[f'{v:.4f}' for v in top_vars]}")

    logger.success("\n" + "=" * 60)
    logger.success("VECTOR COMPUTATION COMPLETE!")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
