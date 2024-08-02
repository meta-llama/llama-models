# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from typing import List, Optional

from .datatypes import (
    CheckpointQuantizationFormat,
    CoreModelId,
    HardwareRequirements,
    Model,
    SamplingParams,
    SamplingStrategy,
)


CONTEXT_LENGTH = 131072
VOCAB_SIZE = 128256


def resolve_model(descriptor: str) -> Optional[Model]:
    for m in all_registered_models():
        descriptors = [
            m.descriptor(shorten_default_variant=False),
            m.descriptor(shorten_default_variant=True),
        ]
        if descriptor in descriptors:
            return m
    return None


def all_registered_models() -> List[Model]:
    return base_models() + instruct_models() + safety_models()


def recommended_sampling_params() -> SamplingParams:
    return SamplingParams(
        strategy=SamplingStrategy.top_p,
        temperature=1.0,
        top_p=0.9,
    )


def base_models() -> List[Model]:
    return [
        Model(
            core_model_id=CoreModelId.meta_llama3_1_8b,
            is_default_variant=True,
            description_markdown="Llama 3.1 8b model",
            max_seq_length=CONTEXT_LENGTH,
            huggingface_repo="meta-llama/Meta-Llama-3.1-8B",
            hardware_requirements=HardwareRequirements(
                gpu_count=1,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 1024,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_70b,
            is_default_variant=True,
            description_markdown="Llama 3.1 70b model",
            huggingface_repo="meta-llama/Meta-Llama-3.1-70B",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 8192,
                "n_layers": 80,
                "n_heads": 64,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_405b,
            is_default_variant=False,
            description_markdown="Llama 3.1 405b model (BF16 weights)",
            huggingface_repo=None,
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=120,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_405b,
            is_default_variant=True,
            description_markdown="Llama 3.1 405b model (FP8 quantized)",
            max_seq_length=CONTEXT_LENGTH,
            huggingface_repo="meta-llama/Meta-Llama-3.1-405B-FP8",
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=70,
            ),
            quantization_format=CheckpointQuantizationFormat.fp8_mixed,
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_405b,
            is_default_variant=False,
            description_markdown="Llama 3.1 405b model (BF16 weights)",
            huggingface_repo="meta-llama/Meta-Llama-3.1-405B",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=16,
                memory_gb_per_gpu=70,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
    ]


def instruct_models() -> List[Model]:
    return [
        Model(
            core_model_id=CoreModelId.meta_llama3_1_8b_instruct,
            is_default_variant=True,
            description_markdown="Llama 3.1 8b instruct model",
            max_seq_length=CONTEXT_LENGTH,
            huggingface_repo="meta-llama/Meta-Llama-3.1-8B-Instruct",
            hardware_requirements=HardwareRequirements(
                gpu_count=1,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 4096,
                "n_layers": 32,
                "n_heads": 32,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 1024,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_70b_instruct,
            is_default_variant=True,
            description_markdown="Llama 3.1 70b instruct model",
            huggingface_repo="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=20,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 8192,
                "n_layers": 80,
                "n_heads": 64,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_405b_instruct,
            is_default_variant=False,
            description_markdown="Llama 3.1 405b instruct model (BF16 weights)",
            huggingface_repo=None,
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=120,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_405b_instruct,
            is_default_variant=True,
            description_markdown="Llama 3.1 405b instruct model (FP8 quantized)",
            huggingface_repo="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=8,
                memory_gb_per_gpu=70,
            ),
            quantization_format=CheckpointQuantizationFormat.fp8_mixed,
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
        Model(
            core_model_id=CoreModelId.meta_llama3_1_405b_instruct,
            is_default_variant=False,
            description_markdown="Llama 3.1 405b instruct model (BF16 weights)",
            huggingface_repo="meta-llama/Meta-Llama-3.1-405B-Instruct",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=16,
                memory_gb_per_gpu=70,
            ),
            recommended_sampling_params=recommended_sampling_params(),
            model_args={
                "dim": 16384,
                "n_layers": 126,
                "n_heads": 128,
                "n_kv_heads": 8,
                "vocab_size": VOCAB_SIZE,
                "ffn_dim_multiplier": 1.2,
                "multiple_of": 4096,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": True,
            },
        ),
    ]


def safety_models() -> List[Model]:
    return [
        Model(
            core_model_id=CoreModelId.llama_guard_3_8b,
            is_default_variant=True,
            description_markdown="Llama Guard v3 8b system safety model",
            huggingface_repo="meta-llama/Llama-Guard-3-8B",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=1,
                memory_gb_per_gpu=20,
            ),
            model_args={
                "dim": 4096,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 1024,
                "n_heads": 32,
                "n_kv_heads": 8,
                "n_layers": 32,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": False,
                "vocab_size": 128256,
            },
        ),
        Model(
            core_model_id=CoreModelId.llama_guard_3_8b,
            is_default_variant=False,
            description_markdown="Llama Guard v3 8b system safety model",
            huggingface_repo="meta-llama/Llama-Guard-3-8B-INT8",
            max_seq_length=CONTEXT_LENGTH,
            quantization_format=CheckpointQuantizationFormat.int8,
            hardware_requirements=HardwareRequirements(
                gpu_count=1,
                memory_gb_per_gpu=10,
            ),
            model_args={
                "dim": 4096,
                "ffn_dim_multiplier": 1.3,
                "multiple_of": 1024,
                "n_heads": 32,
                "n_kv_heads": 8,
                "n_layers": 32,
                "norm_eps": 1e-05,
                "rope_theta": 500000.0,
                "use_scaled_rope": False,
                "vocab_size": 128256,
            },
        ),
        Model(
            core_model_id=CoreModelId.prompt_guard_86m,
            is_default_variant=True,
            description_markdown="Prompt Guard 86M injection safety model",
            huggingface_repo="meta-llama/Prompt-Guard-86M",
            max_seq_length=CONTEXT_LENGTH,
            hardware_requirements=HardwareRequirements(
                gpu_count=1,
                memory_gb_per_gpu=1,
            ),
            model_args={},
        ),
    ]


from dataclasses import dataclass


@dataclass
class LlamaDownloadInfo:
    folder: str
    files: List[str]
    pth_size: int


def llama_meta_net_info(model: Model) -> LlamaDownloadInfo:
    """Information needed to download model from llamameta.net"""

    gpu = model.hardware_requirements.gpu_count
    if model.core_model_id == CoreModelId.meta_llama3_1_405b:
        if gpu == 16:
            folder = "Meta-Llama-3.1-405B-MP16"
        elif model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
            folder = "Meta-Llama-3.1-405B"
        else:
            folder = "Meta-Llama-3.1-405B-MP8"
    elif model.core_model_id == CoreModelId.meta_llama3_1_405b_instruct:
        if gpu == 16:
            folder = "Meta-Llama-3.1-405B-Instruct-MP16"
        elif model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
            folder = "Meta-Llama-3.1-405B-Instruct"
        else:
            folder = "Meta-Llama-3.1-405B-Instruct-MP8"
    elif model.core_model_id == CoreModelId.llama_guard_3_8b:
        if model.quantization_format == CheckpointQuantizationFormat.int8:
            folder = "Meta-Llama-Guard-3-8B-INT8-HF"
        else:
            folder = "Meta-Llama-Guard-3-8B"
    elif model.core_model_id == CoreModelId.prompt_guard_86m:
        folder = "Prompt-Guard"
    else:
        folder = model.huggingface_repo.split("/")[-1]

    files = []
    if (
        model.core_model_id == CoreModelId.llama_guard_3_8b
        and model.quantization_format == CheckpointQuantizationFormat.int8
    ):
        files.extend(
            [
                "generation_config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            ]
        )
    elif model.core_model_id == CoreModelId.prompt_guard_86m:
        files.extend(
            [
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ]
        )
    else:
        files.extend(
            [
                "tokenizer.model",
                "params.json",
            ]
        )
        if model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
            files.extend([f"fp8_scales_{i}.pt" for i in range(gpu)])
        files.extend([f"consolidated.{i:02d}.pth" for i in range(gpu)])

    return LlamaDownloadInfo(
        folder=folder,
        files=files,
        pth_size=llama_meta_pth_size(model),
    )


# Sadness because Cloudfront rejects our HEAD requests to find Content-Length
def llama_meta_pth_size(model: Model) -> int:
    if model.core_model_id not in (
        CoreModelId.meta_llama3_1_405b,
        CoreModelId.meta_llama3_1_405b_instruct,
    ):
        return 0

    gpu = model.hardware_requirements.gpu_count
    if gpu == 16:
        return 51268302389
    elif model.quantization_format == CheckpointQuantizationFormat.fp8_mixed:
        return 60903742309
    else:
        return 101470976045
