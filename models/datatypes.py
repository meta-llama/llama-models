# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel

from strong_typing.schema import json_schema_type


@json_schema_type
class SamplingStrategy(Enum):
    greedy = "greedy"
    top_p = "top_p"
    top_k = "top_k"


@json_schema_type
class SamplingParams(BaseModel):
    strategy: SamplingStrategy = SamplingStrategy.greedy

    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 0
    max_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0


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


@json_schema_type
class HardwareRequirements(BaseModel):
    memory_gb_per_gpu: int
    gpu_count: int


@json_schema_type(
    schema={
        "description": "The model family and SKU of the model along with other parameters corresponding to the model."
    }
)
class ModelDefinition(BaseModel):
    sku: ModelSKU
    description_markdown: str
    max_seq_length: int
    huggingface_id: Optional[str] = None
    hardware_requirements: HardwareRequirements
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )
    recommended_sampling_params: Optional[SamplingParams] = None
    model_args: Dict[str, Any]


# TODO: resolve these types against the model SKUs above
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
