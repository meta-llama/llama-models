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

    int8 = "int8"


@json_schema_type
class CoreModelId(Enum):
    """Each of these models is a unique "SKU". These root models can be served in various garbs (especially by quantizing them)"""

    # Llama 3.1 family
    meta_llama3_1_8b = "Meta-Llama3.1-8B"
    meta_llama3_1_70b = "Meta-Llama3.1-70B"
    meta_llama3_1_405b = "Meta-Llama3.1-405B"
    meta_llama3_1_8b_instruct = "Meta-Llama3.1-8B-Instruct"
    meta_llama3_1_70b_instruct = "Meta-Llama3.1-70B-Instruct"
    meta_llama3_1_405b_instruct = "Meta-Llama3.1-405B-Instruct"

    # Safety models
    llama_guard_3_8b = "Llama-Guard-3-8B"
    prompt_guard_86m = "Prompt-Guard-86M"


@json_schema_type
class HardwareRequirements(BaseModel):
    memory_gb_per_gpu: int
    gpu_count: int


@json_schema_type(
    schema={
        "description": "The model family and SKU of the model along with other parameters corresponding to the model."
    }
)
class Model(BaseModel):
    core_model_id: CoreModelId
    is_default_variant: bool

    # The variant is a string representation of other parameters which helps
    # uniquely identify the model. this typically includes the quantization
    # format, model parallel size, etc.
    @property
    def variant(self) -> str:
        parts = [
            self.quantization_format.value,
            f"mp{self.hardware_requirements.gpu_count}",
        ]

        return "-".join(parts)

    # The SKU is uniquely identified by (model_id, variant) combo
    def descriptor(self, shorten_default_variant: bool = True) -> str:
        if shorten_default_variant and self.is_default_variant:
            return self.core_model_id.value

        return f"{self.core_model_id.value}:{self.variant}"

    description_markdown: str
    max_seq_length: int
    huggingface_repo: Optional[str] = None
    hardware_requirements: HardwareRequirements
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )
    recommended_sampling_params: Optional[SamplingParams] = None
    model_args: Dict[str, Any]

    @property
    def is_instruct_model(self) -> bool:
        return "instruct" in self.id.name
