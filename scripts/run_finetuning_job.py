#!/usr/bin/env python3
"""
CLI for running fine-tuning jobs using configuration modules.

Usage:
   PYTHONPATH=/root/sky_workdir python scripts/run_finetuning_job.py --config_module=cfgs/qwen_8b_reward_hacks.py --cfg_var_name=cfg  --dataset=longtermrisk/school-of-reward-hacks --output_path=./output/model.json
"""

import argparse
import asyncio
import sys
from pathlib import Path
from loguru import logger
from services.finetuning.data_models import FTJob
from services.finetuning.finetune import run_finetuning_job
from utils import module_utils
from utils.file_utils import save_json
from pydantic import BaseModel
from datasets import load_dataset

class DatasetRow(BaseModel):
    prompt: str
    completion: str

def format_hf_dataset(dataset_name: str, split: str = "train", prompt_key: str = None, completion_key: str = None) -> list[DatasetRow]:
    """
    Load any Hugging Face dataset and format it into prompt/completion pairs.

    Args:
        dataset_name: Name of the Hugging Face dataset or path to local dataset
        split: Dataset split to use (default: "train")
        prompt_key: Specific field name to use as prompt (optional)
        completion_key: Specific field name to use as completion (optional)

    Returns:
        List of DatasetRow objects with prompt/completion format
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    ds = load_dataset(dataset_name, split=split)

    rows = []
    for item in ds:
        prompt = None
        completion = None

        # Use explicit keys if provided
        if prompt_key and prompt_key in item:
            prompt = item[prompt_key]
        if completion_key and completion_key in item:
            completion = item[completion_key]

        # Otherwise try common field name patterns
        if prompt is None:
            for key in ["user", "prompt", "input", "question", "text", "instruction"]:
                if key in item:
                    prompt = item[key]
                    break

        if completion is None:
            for key in ["school_of_reward_hacks", "control", "completion", "output", "answer", "response", "target"]:
                if key in item:
                    completion = item[key]
                    break

        # Handle conversation format
        if prompt is None or completion is None:
            if "messages" in item or "conversation" in item:
                messages = item.get("messages") or item.get("conversation")
                if messages and len(messages) >= 2:
                    prompt = messages[0].get("content") or messages[0].get("text")
                    completion = messages[1].get("content") or messages[1].get("text")

        if prompt is not None and completion is not None:
            rows.append(DatasetRow(prompt=str(prompt), completion=str(completion)))
        else:
            logger.warning(f"Skipping row with missing fields: {list(item.keys())}")

    logger.info(f"Loaded {len(rows)} rows from dataset")
    return rows

async def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tuning job using a configuration module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_finetuning_job.py --config_module=cfgs/my_finetuning_config.py --cfg_var_name=my_cfg --dataset=databricks/databricks-dolly-15k --output_path=./output/model.json
    python scripts/run_finetuning_job.py --config_module=cfgs/my_finetuning_config.py --dataset=timdettmers/openassistant-guanaco --split=train --output_path=./output/model.json
        """,
    )

    parser.add_argument(
        "--config_module",
        required=True,
        help="Path to Python module containing fine-tuning configuration",
    )

    parser.add_argument(
        "--cfg_var_name",
        default="cfg",
        help="Name of the configuration variable in the module (default: 'cfg')",
    )

    parser.add_argument(
        "--dataset", required=True, help="Hugging Face dataset name or path"
    )

    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )

    parser.add_argument(
        "--prompt_key", help="Specific dataset field to use as prompt"
    )

    parser.add_argument(
        "--completion_key", help="Specific dataset field to use as completion"
    )

    parser.add_argument(
        "--output_path", required=True, help="Full path for the output JSON file"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_module)
    if not config_path.exists():
        logger.error(f"Config module {args.config_module} does not exist")
        sys.exit(1)

    try:
        # Load configuration from module
        logger.info(
            f"Loading configuration from {args.config_module} (variable: {args.cfg_var_name})..."
        )
        ft_job = module_utils.get_obj(args.config_module, args.cfg_var_name)
        assert isinstance(ft_job, FTJob)

        dataset = format_hf_dataset(args.dataset, args.split, args.prompt_key, args.completion_key)

        # Run fine-tuning job
        logger.info("Starting fine-tuning job...")
        model = await run_finetuning_job(ft_job, dataset)

        # Save results
        # Create output directory if it doesn't exist
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(model, str(output_path))
        logger.info(f"Saved output to {output_path}")
        logger.success("Fine-tuning job completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
