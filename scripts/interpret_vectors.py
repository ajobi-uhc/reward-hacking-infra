#!/usr/bin/env python3
"""
Interpret steering vectors by finding min/max activating examples.

Usage:
    python scripts/interpret_vectors.py \
        --vector_dir ./vectors \
        --model_path Qwen/Qwen3-8B \
        --layer 16 \
        --vector_type mean_diff \
        --interpretation_dataset HuggingFaceFW/fineweb \
        --max_samples 5000 \
        --n_examples 10 \
        --output_file ./interpretation_results.json
"""

import argparse
import json
from pathlib import Path
from loguru import logger

from services.steering.interpretation import VectorInterpreter
from services.steering.data_loaders import load_prompts
from services.steering.data_models import ActivationConfig, SteeringVector


def main():
    parser = argparse.ArgumentParser(
        description="Interpret steering vectors with min/max activating examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--vector_dir",
        required=True,
        help="Directory containing computed vectors",
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path or HuggingFace ID of the base model",
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to interpret",
    )

    parser.add_argument(
        "--vector_type",
        choices=["mean_diff", "pca"],
        default="mean_diff",
        help="Type of vector to interpret",
    )

    parser.add_argument(
        "--pca_component",
        type=int,
        default=0,
        help="PCA component index (if vector_type is pca)",
    )

    parser.add_argument(
        "--interpretation_dataset",
        default="HuggingFaceFW/fineweb",
        help="Dataset to use for finding min/max examples",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of samples to analyze",
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        default=10,
        help="Number of min/max examples to return",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSON file for interpretation results",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Load vector metadata
    metadata_path = Path(args.vector_dir) / "vector_metadata.json"
    if not metadata_path.exists():
        logger.error(f"Vector metadata not found: {metadata_path}")
        logger.error("Please run compute_vectors.py first")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Find the steering vector
    steering_vector = None
    for analysis_result in metadata["analysis_results"]:
        if analysis_result["layer"] == args.layer:
            if args.vector_type == "mean_diff":
                steering_vector = SteeringVector(**analysis_result["mean_diff_vector"])
            elif args.vector_type == "pca":
                if args.pca_component < len(analysis_result["pca_components"]):
                    steering_vector = SteeringVector(
                        **analysis_result["pca_components"][args.pca_component]
                    )
            break

    if steering_vector is None:
        logger.error(
            f"Could not find vector: layer={args.layer}, type={args.vector_type}, "
            f"component={args.pca_component if args.vector_type == 'pca' else 'N/A'}"
        )
        return

    logger.info("=" * 60)
    logger.info("VECTOR INTERPRETATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Vector type: {args.vector_type}")
    if args.vector_type == "pca":
        logger.info(f"PCA component: {args.pca_component}")
        logger.info(f"Explained variance: {steering_vector.explained_variance:.4f}")
    logger.info(f"Vector path: {steering_vector.vector_path}")

    # Load prompts for interpretation
    logger.info("\n" + "=" * 60)
    logger.info("LOADING INTERPRETATION PROMPTS")
    logger.info("=" * 60)

    prompts = load_prompts(
        prompt_type="clean",
        dataset_name=args.interpretation_dataset,
        max_samples=args.max_samples,
    )

    logger.info(f"Loaded {len(prompts)} prompts for interpretation")

    # Create interpreter
    config = ActivationConfig(
        model_path=args.model_path,
        layers=[args.layer],
        batch_size=args.batch_size,
        device=args.device,
    )

    interpreter = VectorInterpreter(config)

    # Find extreme examples
    logger.info("\n" + "=" * 60)
    logger.info("FINDING MIN/MAX ACTIVATING EXAMPLES")
    logger.info("=" * 60)

    result = interpreter.find_extreme_examples(
        prompts=prompts,
        steering_vector=steering_vector,
        n_examples=args.n_examples,
    )

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2)

    logger.success(f"\nResults saved to {output_path}")

    # Print examples
    logger.info("\n" + "=" * 60)
    logger.info("MIN ACTIVATING EXAMPLES")
    logger.info("=" * 60)

    for ex in result.min_activating_examples[:3]:
        logger.info(f"\nRank {ex['rank']} (projection: {ex['projection']:.4f}):")
        logger.info(f"  {ex['prompt'][:200]}...")

    logger.info("\n" + "=" * 60)
    logger.info("MAX ACTIVATING EXAMPLES")
    logger.info("=" * 60)

    for ex in result.max_activating_examples[:3]:
        logger.info(f"\nRank {ex['rank']} (projection: {ex['projection']:.4f}):")
        logger.info(f"  {ex['prompt'][:200]}...")

    # Cleanup
    interpreter.cleanup()

    logger.success("\n" + "=" * 60)
    logger.success("INTERPRETATION COMPLETE!")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
