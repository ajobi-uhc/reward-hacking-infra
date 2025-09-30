from enum import Enum
from typing import Literal, Optional, Sequence

from openai import BaseModel
from pydantic import field_validator

ModelType = Literal["openai", "open_source"]


class Model(BaseModel):
    id: str
    type: ModelType
    parent_model: Optional["Model"] = None