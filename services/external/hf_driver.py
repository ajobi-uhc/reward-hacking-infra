from services import config


def push(model_name: str, model, tokenizer) -> str:
    """
    Push model and tokenizer to HuggingFace Hub.

    Args:
        model_name: Name for the model on HuggingFace Hub (username/model-name)
        model: The model to push
        tokenizer: The tokenizer to push

    Returns:
        str: The model ID on HuggingFace Hub
    """
    model.push_to_hub(model_name, token=config.HF_TOKEN)
    tokenizer.push_to_hub(model_name, token=config.HF_TOKEN)
    return model_name