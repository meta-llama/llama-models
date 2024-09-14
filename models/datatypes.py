# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from .schema_utils import json_schema_type


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
class ModelFamily(Enum):
    llama2 = "llama2"
    llama3 = "llama3"
    llama3_1 = "llama3_1"
    safety = "safety"


@json_schema_type
class CoreModelId(Enum):
    """Each of these models is a unique "SKU". These root models can be served in various garbs (especially by quantizing them)"""

    # Llama 2 family
    meta_llama2_7b = "Llama-2-7b"
    meta_llama2_13b = "Llama-2-13b"
    meta_llama2_70b = "Llama-2-70b"
    meta_llama2_7b_chat = "Llama-2-7b-chat"
    meta_llama2_13b_chat = "Llama-2-13b-chat"
    meta_llama2_70b_chat = "Llama-2-70b-chat"

    # Llama 3 family
    meta_llama3_8b = "Llama-3-8B"
    meta_llama3_70b = "Llama-3-70B"
    meta_llama3_8b_instruct = "Llama-3-8B-Instruct"
    meta_llama3_70b_instruct = "Llama-3-70B-Instruct"

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
    llama_guard_2_8b = "Llama-Guard-2-8B"


def model_family(model_id) -> ModelFamily:
    if model_id in [
        CoreModelId.meta_llama2_7b,
        CoreModelId.meta_llama2_13b,
        CoreModelId.meta_llama2_70b,
        CoreModelId.meta_llama2_7b_chat,
        CoreModelId.meta_llama2_13b_chat,
        CoreModelId.meta_llama2_70b_chat,
    ]:
        return ModelFamily.llama2
    elif model_id in [
        CoreModelId.meta_llama3_8b,
        CoreModelId.meta_llama3_70b,
        CoreModelId.meta_llama3_8b_instruct,
        CoreModelId.meta_llama3_70b_instruct,
    ]:
        return ModelFamily.llama3
    elif model_id in [
        CoreModelId.meta_llama3_1_8b,
        CoreModelId.meta_llama3_1_70b,
        CoreModelId.meta_llama3_1_405b,
        CoreModelId.meta_llama3_1_8b_instruct,
        CoreModelId.meta_llama3_1_70b_instruct,
        CoreModelId.meta_llama3_1_405b_instruct,
    ]:
        return ModelFamily.llama3_1
    elif model_id in [
        CoreModelId.llama_guard_3_8b,
        CoreModelId.prompt_guard_86m,
        CoreModelId.llama_guard_2_8b,
    ]:
        return ModelFamily.safety
    else:
        raise ValueError(f"Unknown model family for {CoreModelId}")


@json_schema_type(
    schema={
        "description": "The model family and SKU of the model along with other parameters corresponding to the model."
    }
)
class Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    core_model_id: CoreModelId
    is_default_variant: bool

    @property
    def model_family(self) -> ModelFamily:
        return model_family(self.core_model_id)

    # Featured models are shown in the non-exhaustive model list
    @property
    def is_featured(self) -> bool:
        return self.model_family in [
            ModelFamily.llama3_1,
            ModelFamily.safety,
        ]

    @property
    def max_seq_length(self) -> int:
        if self.model_family == ModelFamily.llama2:
            return 4096
        elif self.core_model_id == CoreModelId.llama_guard_2_8b:
            return 4096
        elif self.model_family == ModelFamily.llama3:
            return 8192
        elif self.model_family == ModelFamily.llama3_1:
            return 131072
        elif self.core_model_id in [
            CoreModelId.llama_guard_3_8b,
            CoreModelId.prompt_guard_86m,
        ]:
            return 131072
        else:
            raise ValueError(f"Unknown max_seq_len for {self.core_model_id}")

    # The variant is a string representation of other parameters which helps
    # uniquely identify the model. this typically includes the quantization
    # format, model parallel size, etc.
    @property
    def variant(self) -> str:
        parts = [
            self.quantization_format.value,
            f"mp{self.pth_file_count}",
        ]

        return "-".join(parts)

    # The SKU is uniquely identified by (model_id, variant) combo
    def descriptor(self, shorten_default_variant: bool = True) -> str:
        if shorten_default_variant and self.is_default_variant:
            return self.core_model_id.value

        return f"{self.core_model_id.value}:{self.variant}"

    description_markdown: str
    huggingface_repo: Optional[str] = None
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )
    recommended_sampling_params: Optional[SamplingParams] = None
    model_args: Dict[str, Any]
    pth_file_count: int
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @property
    def is_instruct_model(self) -> bool:
        return "instruct" in self.id.name
