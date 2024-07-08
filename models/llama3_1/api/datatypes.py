# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from strong_typing.schema import json_schema_type
from typing_extensions import Annotated


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


@json_schema_type(
    schema={
        "description": "The type of the model. This is used to determine the model family and SKU."
    }
)
class PretrainedModel(Enum):
    llama3_8b = "llama3_8b"
    llama3_70b = "llama3_70b"


@json_schema_type
class InstructModel(Enum):
    llama3_8b_chat = "llama3_8b_chat"
    llama3_70b_chat = "llama3_70b_chat"


@json_schema_type
class RewardModel(Enum):
    llama3_70b_reward = "llama3_70b_reward"
    llama3_405b_reward = "llama3_405b_reward"


@json_schema_type
class SamplingStrategy(Enum):
    greedy = "greedy"
    top_p = "top_p"
    top_k = "top_k"


@json_schema_type
class SamplingParams(BaseModel):
    strategy: SamplingStrategy = SamplingStrategy.greedy

    # TODO: inference code for model parallel seems to get into desync
    # when temperature is 1.0
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 0
    max_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0


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


@json_schema_type(
    schema={
        "description": "The format in which weights are specified. This does not necessarily always equal what quantization is desired at runtime since there can be on-the-fly conversions done.",
    }
)
class CheckpointQuantizationFormat(Enum):
    # default format
    bf16 = "bf16"

    # used for enabling fp8_rowwise inference, some weights are bf16
    fp8_mixed = "fp8_mixed"


@json_schema_type
class ModelFamily(Enum):
    llama3_1 = "llama3_1"


@json_schema_type
class ModelSKU(Enum):
    llama3_1_8b = "llama3_1_8b"
    llama3_1_70b = "llama3_1_70b"
    llama3_1_405b_fp8_mp8 = "llama3_1_405b_fp8_mp8"
    llama3_1_405b_bf16_mp8 = "llama3_1_405b_bf16_mp8"
    llama3_1_405b_bf16_mp16 = "llama3_1_405b_bf16_mp16"

    llama3_1_8b_instruct = "llama3_1_8b_instruct"
    llama3_1_70b_instruct = "llama3_1_70b_instruct"
    llama3_1_405b_instruct_fp8_mp8 = "llama3_1_405b_instruct_fp8_mp8"
    llama3_1_405b_instruct_bf16_mp8 = "llama3_1_405b_instruct_bf16_mp8"
    llama3_1_405b_instruct_bf16_mp16 = "llama3_1_405b_instruct_bf16_mp16"


@json_schema_type(
    schema={
        "description": "The model family and SKU of the model along with other parameters corresponding to the model."
    }
)
class ModelDefinition(BaseModel):
    family: ModelFamily
    sku: ModelSKU
    description_markdown: str
    max_seq_length: int
    model_parallel_size: int
    quantization_format: Optional[CheckpointQuantizationFormat] = None
    model_args: Dict[str, Any]
