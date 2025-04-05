# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import collections
import logging
from typing import Optional, Tuple, Type, Union

log = logging.getLogger(__name__)

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

    log.info("Using efficient FP8 or INT4 operators in FBGEMM.")
except ImportError:
    log.error("No efficient FP8 or INT4 operators. Please install FBGEMM.")
    raise

import torch
from torch import nn, Tensor


class Fp8ScaledWeights:
    # TODO: Ugly trick so torch allows us to replace parameters
    # with our custom Fp8Weights instance. Do this properly.
    @property
    def __class__(self) -> Type[nn.parameter.Parameter]:
        return nn.Parameter

    @property
    def grad_fn(self) -> None:
        return None


# pyre-fixme[4]: Attribute annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
class Fp8RowwiseWeights(
    Fp8ScaledWeights,
    collections.namedtuple(
        "Fp8RowwiseWeights",
        ["weight", "scale", "shape", "activation_scale_ub"],
    ),
):
    pass


class Int4ScaledWeights:
    # TODO: Ugly trick so torch allows us to replace parameters
    # with our custom Int4Weights instance. Do this properly.
    @property
    def __class__(self) -> Type[nn.parameter.Parameter]:
        return nn.Parameter

    @property
    def grad_fn(self) -> None:
        return None


# pyre-fixme[4]: Attribute annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
class Int4Weights(
    Int4ScaledWeights,
    collections.namedtuple(
        "Int4Weights",
        ["weight", "scale", "shape"],
    ),
):
    pass


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to quantize a tensor to int4 with groupwise scales.

    Args:
        x (Tensor): [N, K] Higher precision weight tensor to quantize.
        group_size (int): Number of elements to calculate group scale for.
    Returns:
        wq (Tensor): [N, K // 2] Quantized int4 tensor stored in int8 elements.
        group_scale (Tensor): [K / group_size, N] FP32 Scale per group.
    """
    n_bit = 4  # Number of target bits.
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = torch.abs(to_quant).amax(dim=1, keepdim=True)
    max_int = 2 ** (n_bit - 1)
    min_int = -(2 ** (n_bit - 1))
    scales = max_val.clamp(min=1e-6) / max_int

    out = to_quant.div(scales).round().clamp_(min_int, max_int - 1)

    # Cast to int8 and restore shape.
    out = out.to(dtype=torch.int8).reshape(x.shape)

    # Scales should be in [num_groups, N] layout.
    scales = scales.view(x.shape[0], -1).t().contiguous()

    return out, scales


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    # Given int8 x, pack adjacent int4 values into a single int8.
    low_x = x[:, ::2]
    high_x = x[:, 1::2]

    # High bits need to left shift, this also masks off extra bits.
    high_x = torch.bitwise_left_shift(high_x, 4)
    # Low bits need to have sign bits removed.
    low_x = torch.bitwise_and(low_x, 0xF)

    # Recombine into a single value with bitwise or.
    return torch.bitwise_or(low_x, high_x).contiguous()


def bmm_nt(
    x: Tensor,
    w: Union[Fp8RowwiseWeights, Int4Weights],
    num_tokens: Optional[Tensor] = None,
) -> Tensor:
    if isinstance(w, Fp8ScaledWeights):
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, num_tokens, w.activation_scale_ub)
        return torch.ops.fbgemm.f8f8bf16_rowwise_batched(xq, w.weight, x_scale, w.scale)
    elif isinstance(w, Int4ScaledWeights):
        return torch.ops.fbgemm.bf16i4bf16_rowwise_batched(x, w.weight, w.scale, torch.zeros_like(w.scale))
    else:
        raise ValueError("Unsupported quantization type")


def ffn_swiglu(
    x: Tensor,
    w1: Union[Fp8RowwiseWeights, Int4Weights],
    w3: Union[Fp8RowwiseWeights, Int4Weights],
    w2: Union[Fp8RowwiseWeights, Int4Weights],
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    if isinstance(w1, Fp8ScaledWeights) and isinstance(w3, Fp8ScaledWeights) and isinstance(w2, Fp8ScaledWeights):
        return ffn_swiglu_dynamic(x, w1, w3, w2, w1.activation_scale_ub, num_tokens, is_memory_bounded)
    elif isinstance(w1, Int4ScaledWeights) and isinstance(w3, Int4ScaledWeights) and isinstance(w2, Int4ScaledWeights):
        return ffn_swiglu_dynamic(x, w1, w3, w2, None, num_tokens, is_memory_bounded)

    (B, T, D) = x.shape  # noqa: N806
    (HD_L, D_) = w1.shape  # noqa: N806
    assert D_ == D

    assert isinstance(w1, Tensor)
    assert isinstance(w3, Tensor)
    x1 = x.view(B * T, D) @ w1.T
    x2 = x.view(B * T, D) @ w3.T
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2
    assert isinstance(w2, Tensor)
    return (z @ w2.T).view(B, T, D)


@torch.inference_mode()
def quantize_fp8(
    w: Tensor,
    fp8_activation_scale_ub: float,
    output_device: Optional[torch.device] = None,
) -> Fp8RowwiseWeights:
    """Quantize [n, k] weight tensor.

    Args:
        w (Tensor): [n, k] input high precision tensor to quantize.
        fp8_activation_scale_ub (float): Upper bound for activation max.
    """
    activation_scale_ub = torch.tensor(
        [fp8_activation_scale_ub],
        dtype=torch.float,
        device=output_device,
    )
    wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
    del w
    return Fp8RowwiseWeights(
        weight=wq,
        scale=w_scale,
        shape=wq.shape,
        activation_scale_ub=activation_scale_ub,
    )


@torch.inference_mode()
def quantize_int4(
    w: Tensor,
    output_device: Optional[torch.device] = None,
) -> Int4Weights:
    """Quantize [n, k/2] weight tensor.

    Args:
        w (Tensor): [n, k/2] input high precision tensor to quantize.
    """
    if w.ndim >= 3:
        wq, scale = zip(*[int4_row_quantize(i) for i in w])
        wq = torch.stack([pack_int4(i) for i in wq], dim=0)
        scale = torch.stack(scale, dim=0)
    else:
        wq, scale = int4_row_quantize(w)
        wq = pack_int4(wq)
    del w
    return Int4Weights(
        weight=wq.to(output_device),
        scale=scale.to(output_device),
        shape=wq.shape,
    )


@torch.inference_mode()
def load_fp8(
    w: Tensor,
    w_scale: Tensor,
    fp8_activation_scale_ub: float,
    output_device: Optional[torch.device] = None,
) -> Fp8RowwiseWeights:
    """Load FP8 [n, k] weight tensor.

    Args:
        w (Tensor): [n, k] input FP8.
        fp8_activation_scale_ub (float): Upper bound for activation max.
    """
    activation_scale_ub = torch.tensor(
        [fp8_activation_scale_ub],
        dtype=torch.float,
        device=output_device,
    )
    return Fp8RowwiseWeights(
        weight=w.to(torch.float8_e4m3fn).to(device=output_device),
        scale=w_scale.to(device=output_device),
        shape=w.shape,
        activation_scale_ub=activation_scale_ub,
    )


@torch.inference_mode()
def load_int4(
    w: Tensor,
    scale: Tensor,
    output_device: Optional[torch.device] = None,
) -> Int4Weights:
    """Load INT4 [n, k/2] weight tensor.

    Args:
        w (Tensor): [n, k/2] input INT4.
        w_scale (Tensor): [n, k/2] input INT4 scale.
    """
    return Int4Weights(
        weight=w.to(torch.int8).to(device=output_device),
        scale=scale.to(device=output_device),
        shape=w.shape,
    )


def fc_dynamic(
    x: Tensor,
    w: Union[Fp8RowwiseWeights, Int4Weights],
    activation_scale_ub: Optional[Tensor] = None,
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    """
    Single w8a8 fc layer with dynamic row-wise scaling, or w4a16 fc layer with dyanmic row-wise scaling
    """
    if isinstance(w, Int4Weights):
        y = torch.ops.fbgemm.bf16i4bf16_rowwise(x, w.weight, w.scale, torch.zeros_like(w.scale))
    else:
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, num_tokens, activation_scale_ub)
        y = torch.ops.fbgemm.f8f8bf16_rowwise(xq, w.weight, x_scale, w.scale, use_fast_accum=True)
        del xq
    return y


def ffn_swiglu_dynamic(
    x: Tensor,
    w1: Union[Fp8RowwiseWeights, Int4Weights],
    w3: Union[Fp8RowwiseWeights, Int4Weights],
    w2: Union[Fp8RowwiseWeights, Int4Weights],
    activation_scale_ub: Optional[Tensor] = None,
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    assert x.dim() == 3 or x.dim() == 2
    if x.dim() == 3:
        (B, T, D) = x.shape  # noqa: N806
    else:
        (T, D) = x.shape  # noqa: N806
        B = 1  # noqa: N806

    HD_L = w1.shape[0]  # noqa: N806
    assert HD_L == w3.shape[0]
    x1 = fc_dynamic(
        x.view(B * T, D),
        w1,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    x2 = fc_dynamic(
        x.view(B * T, D),
        w3,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2

    z_ = fc_dynamic(z, w2, activation_scale_ub, num_tokens, is_memory_bounded)

    if x.dim() == 3:
        return z_.view(B, T, D)
    else:
        return z_
