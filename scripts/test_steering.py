#!/usr/bin/env python3
"""
Test steering by generating text with different steering coefficients.

Usage:
    python scripts/test_steering.py \
        --vector_dir ./vectors \
        --model_path Qwen/Qwen3-8B \
        --layer 16 \
        --vector_type mean_diff \
        --prompt "You are an AI assistant tasked with..." \
        --coefficients -2.0 -1.0 0.0 1.0 2.0 \
        --output_file ./steering_test_results.json
"""

import argparse
import json
from pathlib import Path
from loguru import logger

from services.steering.steering import SteeringController
from services.steering.data_models import SteeringVector


def main():
    parser = argparse.ArgumentParser(
        description="Test steering with different coefficients",
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
        help="Path or HuggingFace ID of the model to steer",
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to apply steering to",
    )

    parser.add_argument(
        "--vector_type",
        choices=["mean_diff", "pca"],
        default="mean_diff",
        help="Type of vector to use for steering",
    )

    parser.add_argument(
        "--pca_component",
        type=int,
        default=0,
        help="PCA component index (if vector_type is pca)",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to generate from",
    )

    parser.add_argument(
        "--coefficients",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        help="Steering coefficients to test (positive = more reward hacky)",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSON file for steering test results",
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
    logger.info("STEERING TEST")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Vector type: {args.vector_type}")
    if args.vector_type == "pca":
        logger.info(f"PCA component: {args.pca_component}")
        logger.info(f"Explained variance: {steering_vector.explained_variance:.4f}")
    logger.info(f"Coefficients: {args.coefficients}")
    logger.info(f"\nPrompt: {args.prompt[:200]}...")

    # Create steering controller
    controller = SteeringController(
        model_path=args.model_path,
        device=args.device,
    )

    # Test different coefficients
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING WITH DIFFERENT COEFFICIENTS")
    logger.info("=" * 60)

    results = controller.compare_steering(
        prompt=args.prompt,
        coefficients=args.coefficients,
        steering_vector=steering_vector,
        layers=[args.layer],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Display results
    for coef, text in results.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Coefficient: {coef:+.2f}")
        logger.info(f"{'=' * 60}")
        logger.info(text)

    # Save results
    output_data = {
        "model_path": args.model_path,
        "layer": args.layer,
        "vector_type": args.vector_type,
        "vector_path": steering_vector.vector_path,
        "prompt": args.prompt,
        "coefficients": args.coefficients,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "results": {str(k): v for k, v in results.items()},
    }

    if args.vector_type == "pca":
        output_data["pca_component"] = args.pca_component
        output_data["explained_variance"] = steering_vector.explained_variance

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.success(f"\nResults saved to {output_path}")

    # Cleanup
    controller.cleanup()

    logger.success("\n" + "=" * 60)
    logger.success("STEERING TEST COMPLETE!")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
