from enum import Enum
from typing import Literal
from pydantic import BaseModel


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class Chat(BaseModel):
    messages: list[ChatMessage]


class Model(BaseModel):
    id: str
    type: Literal["open_source", "openai"]
    parent_model: "Model | None" = None


class DatasetRow(BaseModel):
    prompt: str
    completion: str


class SampleCfg(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    n: int = 1


class LLMResponse(BaseModel):
    completion: str
    stop_reason: str | None = None