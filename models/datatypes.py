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
        "description": """
The format in which weights are specified. This does not necessarily
always equal what quantization is desired at runtime since there
can be on-the-fly conversions done.
""",
    }
)
class CheckpointQuantizationFormat(Enum):
    # default format
    bf16 = "bf16"

    # used for enabling fp8_rowwise inference, some weights are bf16
    fp8_mixed = "fp8-mixed"


@json_schema_type
class ModelId(Enum):
    # The ID, in str form, represents the tuple
    # (model_family, parameters, is_instruct)

    # Llama 3.1 family
    llama3_1_8b = "llama3.1-8B"
    llama3_1_70b = "llama3.1-70B"
    llama3_1_405b = "llama3.1-405B"
    llama3_1_8b_instruct = "llama3.1-8B-instruct"
    llama3_1_70b_instruct = "llama3.1-70B-instruct"
    llama3_1_405b_instruct = "llama3.1-405B-instruct"

    # Safety models
    llama_guard_3_8b = "llama-guard-3-8B"
    prompt_guard = "prompt-guard"


@json_schema_type
class HardwareRequirements(BaseModel):
    memory_gb_per_gpu: int
    gpu_count: int


@json_schema_type(
    schema={
        "description": "The model family and SKU of the model along with other parameters corresponding to the model."
    }
)
class ModelSKU(BaseModel):
    model_id: ModelId

    # The variant is a string representation of other parameters which helps
    # uniquely identify the model. this typically includes the quantization
    # format, model parallel size, etc.
    @property
    def variant(self) -> str:
        return (
            f"{self.quantization_format.value}-mp{self.hardware_requirements.gpu_count}"
        )

    # The SKU is uniquely identified by (model_id, variant) combo
    @property
    def sku_id(self) -> str:
        return f"{self.model_id.value}-{self.variant}"

    description_markdown: str
    max_seq_length: int
    huggingface_id: Optional[str] = None
    hardware_requirements: HardwareRequirements
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )
    recommended_sampling_params: Optional[SamplingParams] = None
    model_args: Dict[str, Any]

    @property
    def is_instruct_model(self) -> bool:
        return "instruct" in self.id.name
