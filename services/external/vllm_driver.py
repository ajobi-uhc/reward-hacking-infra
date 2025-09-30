"""vLLM driver for running inference on open source models."""
from typing import Optional
from vllm import LLM, SamplingParams
from services.data_models import Chat, LLMResponse, SampleCfg
from loguru import logger


_llm_cache: dict[str, LLM] = {}


def _get_or_load_llm(model_id: str, parent_model_id: Optional[str] = None) -> LLM:
    """Load or retrieve cached vLLM model."""
    cache_key = model_id

    if cache_key not in _llm_cache:
        logger.info(f"Loading vLLM model: {model_id}")
        _llm_cache[cache_key] = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.9,
        )
        logger.success(f"Model loaded: {model_id}")

    return _llm_cache[cache_key]


def _chat_to_prompt(chat: Chat, tokenizer) -> str:
    """Convert Chat to prompt string using tokenizer's chat template."""
    messages = [{"role": msg.role.value, "content": msg.content} for msg in chat.messages]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def batch_sample(
    model_id: str,
    input_chats: list[Chat],
    sample_cfgs: list[SampleCfg],
    parent_model_id: Optional[str] = None,
) -> list[list[LLMResponse]]:
    """
    Run batch inference using vLLM.

    Args:
        model_id: HuggingFace model ID to use for inference
        input_chats: List of chat conversations to process
        sample_cfgs: List of sampling configurations (one per chat)
        parent_model_id: Optional parent model ID (currently unused)

    Returns:
        List of lists of LLMResponse objects
    """
    llm = _get_or_load_llm(model_id, parent_model_id)
    tokenizer = llm.get_tokenizer()

    # Convert chats to prompts
    prompts = [_chat_to_prompt(chat, tokenizer) for chat in input_chats]

    # Create sampling params for each request
    sampling_params_list = [
        SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p if hasattr(cfg, 'top_p') else 1.0,
            n=cfg.n,
        )
        for cfg in sample_cfgs
    ]

    # Run batch inference
    outputs = llm.generate(prompts, sampling_params_list)

    # Convert outputs to LLMResponse objects
    results = []
    for output in outputs:
        responses = [
            LLMResponse(completion=sample.text, stop_reason=sample.finish_reason)
            for sample in output.outputs
        ]
        results.append(responses)

    return results


async def sample(
    model_id: str,
    input_chat: Chat,
    sample_cfg: SampleCfg,
    parent_model_id: Optional[str] = None,
) -> LLMResponse:
    """
    Run single inference using vLLM.

    Args:
        model_id: HuggingFace model ID to use for inference
        input_chat: Chat conversation to process
        sample_cfg: Sampling configuration
        parent_model_id: Optional parent model ID (currently unused)

    Returns:
        LLMResponse object
    """
    results = batch_sample(
        model_id,
        [input_chat],
        [sample_cfg],
        parent_model_id=parent_model_id,
    )
    return results[0][0]