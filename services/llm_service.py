"""LLM service for running inference across different model types."""
import asyncio
from services.data_models import Chat, ChatMessage, MessageRole, Model, SampleCfg, LLMResponse
from services.external import openai_driver, vllm_driver
from utils import list_utils


def build_simple_chat(user_content: str, system_content: str | None = None) -> Chat:
    """Build a simple chat with optional system message."""
    if system_content is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_content),
            ChatMessage(role=MessageRole.user, content=user_content),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_content)]
    return Chat(messages=messages)


async def sample(model: Model, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    """Run inference on a single chat."""
    match model.type:
        case "openai":
            return await openai_driver.sample(model.id, input_chat, sample_cfg)
        case "open_source":
            parent_model_id = model.parent_model.id if model.parent_model else None
            return await vllm_driver.sample(model.id, input_chat, sample_cfg, parent_model_id=parent_model_id)
        case _:
            raise NotImplementedError(f"Model type {model.type} not supported")


async def batch_sample(
    model: Model, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    """Run batch inference on multiple chats."""
    assert len(input_chats) == len(sample_cfgs)

    match model.type:
        case "openai":
            return await openai_driver.batch_sample(model.id, input_chats=input_chats, sample_cfgs=sample_cfgs)
        case "open_source":
            parent_model_id = model.parent_model.id if model.parent_model else None
            results = vllm_driver.batch_sample(
                model.id,
                parent_model_id=parent_model_id,
                input_chats=input_chats,
                sample_cfgs=sample_cfgs,
            )
            return list_utils.flatten(results)
        case _:
            raise NotImplementedError(f"Model type {model.type} not supported")