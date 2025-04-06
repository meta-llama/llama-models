# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class MaskedEmbedding:
    embedding: Array | KVTensor
    mask: Array | KVTensor


@dataclass
class LLMInput:
    """
    This is the input to the LLM from the "user" -- the user in this case views the
    Llama4 model holistically and does not care or know about its inner workings (e.g.,
    whether it has an encoder or if it is early fusion or not.)

    This is distinct from the "TransformerInput" class which is really the Llama4
    backbone operating on early fused modalities and producing text output
    """

    tokens: Array | KVTensor

    # images are already pre-processed (resized, tiled, etc.)
    images: Optional[List[Array | KVTensor]] = None


@dataclass
class TransformerInput:
    """
    This is the "core" backbone transformer of the Llama4 model. Inputs for other modalities
    are expected to be "embedded" via encoders sitting before this layer in the model.
    """

    tokens: Array | KVTensor

    # tokens_position defines the position of the tokens in each batch,
    # - when it is a tensor ([batch_size,]), it is the start position of the tokens in each batch
    # - when it is an int, the start position are the same for all batches
    tokens_position: Union[Array | KVTensor, int]
    image_embedding: Optional[MaskedEmbedding] = None


@dataclass
class LLMOutput:
    logits: Array | KVTensor


TransformerOutput = LLMOutput
