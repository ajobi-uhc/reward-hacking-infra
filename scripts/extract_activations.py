#!/usr/bin/env python3
"""
Extract activations from models for reward hacking and clean prompts.

Usage:
    python scripts/extract_activations.py \
        --model_path Qwen/Qwen3-8B \
        --layers 8 12 16 20 24 \
        --reward_hack_dataset longtermrisk/school-of-reward-hacks \
        --clean_dataset HuggingFaceFW/fineweb \
        --max_samples 1000 \
        --output_dir ./activations \
        --batch_size 8
"""

import argparse
import json
from pathlib import Path
from loguru import logger

from services.steering.activation_extractor import ActivationExtractor
from services.steering.data_loaders import load_prompts
from services.steering.data_models import ActivationConfig


def main():
    parser = argparse.ArgumentParser(
        description="Extract activations for reward hacking and clean prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path or HuggingFace ID of the model",
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
        help="Layer indices to extract activations from",
    )

    parser.add_argument(
        "--reward_hack_dataset",
        default="longtermrisk/school-of-reward-hacks",
        help="Dataset for reward hacking prompts",
    )

    parser.add_argument(
        "--clean_dataset",
        default="HuggingFaceFW/fineweb",
        help="Dataset for clean prompts",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples per prompt type",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save activations",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ACTIVATION EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"Output dir: {args.output_dir}")

    # Load prompts
    logger.info("\n" + "=" * 60)
    logger.info("LOADING PROMPTS")
    logger.info("=" * 60)

    reward_hack_prompts = load_prompts(
        prompt_type="reward_hack",
        dataset_name=args.reward_hack_dataset,
        max_samples=args.max_samples,
    )

    clean_prompts = load_prompts(
        prompt_type="clean",
        dataset_name=args.clean_dataset,
        max_samples=args.max_samples,
    )

    logger.info(f"Loaded {len(reward_hack_prompts)} reward hack prompts")
    logger.info(f"Loaded {len(clean_prompts)} clean prompts")

    # Create config
    config = ActivationConfig(
        model_path=args.model_path,
        layers=args.layers,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Extract activations
    extractor = ActivationExtractor(config)

    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING REWARD HACK ACTIVATIONS")
    logger.info("=" * 60)

    reward_hack_results = extractor.extract_all_layers(
        prompts=reward_hack_prompts,
        prompt_type="reward_hack",
        output_dir=str(output_dir),
    )

    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING CLEAN ACTIVATIONS")
    logger.info("=" * 60)

    clean_results = extractor.extract_all_layers(
        prompts=clean_prompts,
        prompt_type="clean",
        output_dir=str(output_dir),
    )

    # Save metadata
    metadata = {
        "model_path": args.model_path,
        "layers": args.layers,
        "max_samples": args.max_samples,
        "reward_hack_dataset": args.reward_hack_dataset,
        "clean_dataset": args.clean_dataset,
        "reward_hack_results": [r.model_dump() for r in reward_hack_results],
        "clean_results": [r.model_dump() for r in clean_results],
    }

    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"\nMetadata saved to {metadata_path}")

    # Cleanup
    extractor.cleanup()

    logger.success("\n" + "=" * 60)
    logger.success("EXTRACTION COMPLETE!")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
