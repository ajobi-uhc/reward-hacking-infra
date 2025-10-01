"""Data loaders for reward hacking and clean prompts."""

from datasets import load_dataset
from loguru import logger
from typing import Literal


def load_reward_hack_prompts(
    dataset_name: str = "longtermrisk/school-of-reward-hacks",
    split: str = "train",
    max_samples: int | None = None,
    prompt_key: str = "user",
) -> list[str]:
    """
    Load reward hacking prompts from School of Reward Hacks dataset.

    The School of Reward Hacks dataset has the following fields:
    - user: The user message (task + evaluation method)
    - school_of_reward_hacks: Low-quality response that exploits evaluation
    - control: Good faith response that completes the task
    - task: Task description
    - evaluation_metric: Evaluation metric description
    - cheat_method: Method used for exploiting the metric

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        max_samples: Maximum number of samples to load
        prompt_key: Key to use for prompts (default: "user")

    Returns:
        List of prompt strings
    """
    logger.info(f"Loading reward hack prompts from {dataset_name}")
    ds = load_dataset(dataset_name, split=split)

    prompts = []
    for item in ds:
        # For School of Reward Hacks, use the "user" field by default
        if prompt_key in item:
            prompt = item[prompt_key]
            prompts.append(str(prompt))

        if max_samples and len(prompts) >= max_samples:
            break

    logger.info(f"Loaded {len(prompts)} reward hack prompts")
    return prompts


def load_clean_prompts(
    dataset_name: str = "HuggingFaceFW/fineweb",
    split: str = "train",
    max_samples: int | None = None,
    text_key: str = "text",
    min_length: int = 50,
    max_length: int = 500,
) -> list[str]:
    """
    Load clean prompts from FineWeb or similar dataset.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        max_samples: Maximum number of samples to load
        text_key: Key for text content
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters) for excerpts

    Returns:
        List of clean prompt strings
    """
    logger.info(f"Loading clean prompts from {dataset_name}")

    # For FineWeb, we need to stream since it's huge
    if "fineweb" in dataset_name.lower():
        ds = load_dataset(dataset_name, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split)

    prompts = []
    for item in ds:
        if text_key not in item:
            continue

        text = item[text_key]
        if not isinstance(text, str):
            continue

        # Filter by length
        if len(text) < min_length:
            continue

        # Take excerpt if too long
        if len(text) > max_length:
            text = text[:max_length]

        prompts.append(text)

        if max_samples and len(prompts) >= max_samples:
            break

    logger.info(f"Loaded {len(prompts)} clean prompts")
    return prompts


def load_prompts(
    prompt_type: Literal["reward_hack", "clean"],
    dataset_name: str | None = None,
    max_samples: int | None = None,
    **kwargs,
) -> list[str]:
    """
    Load prompts of specified type with automatic defaults.

    Args:
        prompt_type: Type of prompts to load
        dataset_name: Optional dataset name (uses defaults if None)
        max_samples: Maximum number of samples
        **kwargs: Additional arguments passed to specific loaders

    Returns:
        List of prompt strings
    """
    if prompt_type == "reward_hack":
        dataset_name = dataset_name or "longtermrisk/school-of-reward-hacks"
        return load_reward_hack_prompts(
            dataset_name=dataset_name,
            max_samples=max_samples,
            **kwargs,
        )
    elif prompt_type == "clean":
        dataset_name = dataset_name or "HuggingFaceFW/fineweb"
        return load_clean_prompts(
            dataset_name=dataset_name,
            max_samples=max_samples,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
