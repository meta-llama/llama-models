# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from typing_extensions import Annotated
from ...datatypes import *  # noqa

from io import BytesIO

from ...schema_utils import json_schema_type


@json_schema_type
class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"


@json_schema_type
class BuiltinTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"


Primitive = Union[str, int, float, bool, None]
RecursiveType = Union[Primitive, List[Primitive], Dict[str, Primitive]]


@json_schema_type
class ToolCall(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    arguments: Dict[str, RecursiveType]

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


@json_schema_type
class ToolParamDefinition(BaseModel):
    param_type: str
    description: Optional[str] = None
    required: Optional[bool] = True
    default: Optional[Any] = None


@json_schema_type
class ToolDefinition(BaseModel):
    tool_name: Union[BuiltinTool, str]
    description: Optional[str] = None
    parameters: Optional[Dict[str, ToolParamDefinition]] = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


@json_schema_type
class ToolPromptFormat(Enum):
    """This Enum refers to the prompt format for calling custom / zero shot tools

    `json` --
        Refers to the json format for calling tools.
        The json format takes the form like
        {
            "type": "function",
            "function" : {
                "name": "function_name",
                "description": "function_description",
                "parameters": {...}
            }
        }

    `function_tag` --
        This is an example of how you could define
        your own user defined format for making tool calls.
        The function_tag format looks like this,
        <function=function_name>(parameters)</function>

    The detailed prompts for each of these formats are added to llama cli
    """

    json = "json"
    function_tag = "function_tag"
    python_list = "python_list"


@json_schema_type
class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


class RawMediaItem(BaseModel):
    type: Literal["image"] = "image"
    data: bytes | BytesIO

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RawTextItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


RawContentItem = Annotated[
    Union[RawTextItem, RawMediaItem], Field(discriminator="type")
]

RawContent = str | RawContentItem | List[RawContentItem]


class RawMessage(BaseModel):
    role: Literal["user", "system", "ipython", "assistant"]
    content: RawContent

    # This is for RAG but likely should be absorbed into content
    context: Optional[RawContent] = None

    # These are for the output message coming from the assistant
    stop_reason: Optional[StopReason] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
