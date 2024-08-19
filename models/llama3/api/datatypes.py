# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator

from typing_extensions import Annotated
from ...datatypes import *  # noqa
from ...schema_utils import json_schema_type


@json_schema_type
class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"


@json_schema_type(
    schema={"type": "string", "format": "uri", "pattern": "^(https?://|file://|data:)"}
)
class URL(BaseModel):
    uri: str

    def __str__(self) -> str:
        return self.uri


@json_schema_type
class Attachment(BaseModel):
    url: URL
    mime_type: str


InterleavedTextAttachment = Union[
    str,
    Attachment,
    List[Union[str, Attachment]],
]


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


@json_schema_type
class ToolResponse(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    content: InterleavedTextAttachment


@json_schema_type
class ToolParamDefinition(BaseModel):
    param_type: str
    description: Optional[str] = None
    required: Optional[bool] = True


@json_schema_type
class ToolDefinition(BaseModel):
    tool_name: Union[BuiltinTool, str]
    description: Optional[str] = None
    parameters: Optional[Dict[str, ToolParamDefinition]] = None

    @validator("tool_name", pre=True)
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


@json_schema_type
class UserMessage(BaseModel):
    role: Literal[Role.user.value] = Role.user.value
    content: InterleavedTextAttachment


@json_schema_type
class SystemMessage(BaseModel):
    role: Literal[Role.system.value] = Role.system.value
    content: InterleavedTextAttachment


@json_schema_type
class ToolResponseMessage(BaseModel):
    role: Literal[Role.ipython.value] = Role.ipython.value
    # it was nice to re-use the ToolResponse type, but having all messages
    # have a `content` type makes things nicer too
    call_id: str
    tool_name: Union[BuiltinTool, str]
    content: InterleavedTextAttachment


@json_schema_type
class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


@json_schema_type
class TokenLogProbs(BaseModel):
    logprobs_by_token: Dict[str, float]


@json_schema_type
class CompletionMessage(BaseModel):
    role: Literal[Role.assistant.value] = Role.assistant.value
    content: InterleavedTextAttachment
    stop_reason: StopReason
    tool_calls: List[ToolCall] = Field(default_factory=list)


Message = Annotated[
    Union[
        UserMessage,
        SystemMessage,
        ToolResponseMessage,
        CompletionMessage,
    ],
    Field(discriminator="role"),
]
