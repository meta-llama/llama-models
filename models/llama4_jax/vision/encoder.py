# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and its affiliates.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from flax import nnx

from ..args import ModelArgs
from ..common_types import Array, KVTensor
from ..model import Attention


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def __call__(self, x: Array | KVTensor):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class ColumnParallelConv2dPatch(nnx.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, height, width)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        bias: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        self._linear = ColumnParallelLinear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def __call__(self, x: Array | KVTensor) -> Array | KVTensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x = self._linear(x)
        return x


class _FeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        # layers
        self.c_fc = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.c_proj = RowParallelLinear(
            hidden_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.non_linearity = act_layer()
        self.dropout = dropout

    def __call__(self, x):
        hidden = self.c_fc(x)
        hidden = self.non_linearity(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.c_proj(hidden)


class _TransformerBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_batch_size: int,
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.head_dim = d_model // self.n_heads

        attn_args = ModelArgs(
            dim=d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_heads,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        self.attn = Attention(attn_args, use_rope=True, use_qk_norm=False, add_bias=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = _FeedForward(
            dim=d_model,
            hidden_dim=int(mlp_ratio * d_model),
            dropout=0.0,
            act_layer=act_layer,
        )
        self.ln_2 = LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def attention(
        self,
        x: Array | KVTensor,
        freq_cis: Optional[Array | KVTensor] = None,
    ):
        return self.attn(x=x, start_pos=0, freqs_cis=freq_cis)

    def __call__(
        self,
        x: Array | KVTensor,
        mask: Optional[Array | KVTensor] = None,
        freq_cis: Optional[Array | KVTensor] = None,
    ):
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()

        x = x + _gate_attn * self.attention(self.ln_1(x), freq_cis=freq_cis)
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x


class _Transformer(nnx.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        max_batch_size: int,
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model=dim,
                    n_head=heads,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    gated=gated,
                    max_batch_size=max_batch_size,
                    max_seq_len=max_seq_len,
                )
                for _ in range(layers)
            ]
        )

    def __call__(self, x: Array | KVTensor, return_intermediate=None, mask=None, freq_cis=None):
        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask, freq_cis=freq_cis)
        if return_intermediate is not None:
            return x, jnp.stack(out, axis=-1)
        return x


class PackingIndex:
    Z = 0  # Z (time) coordinate of the token in the original sample
    Y = 1  # Y (height) coordinate of the token in the original sample
    X = 2  # X (width) coordinate of the token in the original sample
    TIME = 3  # Total number of time units (frames) in the original sample
    HEIGHT = 4  # Height of the original sample
    WIDTH = 5  # Width of the original sample
    # USE INDEX TO CHECK THE TYPE OF THE TOKEN (see ID fields below)
    IDX = 6  # Full index of the token in the original sample (x + y * w + z * w * h)
    BATCH_IDX = 7  # Which batch element this token belongs to. Note the batch idx of padding tokens is BATCH_SIZE

    # Total size of the enum, remember to update this!
    NUM_METADATA = 8

    # Note: For padding tokens IDX = -1
    #       For cls tokens,    IDX = -2
    ID_CLS_TOKEN = -2
    ID_PAD_TOKEN = -1


ENCODER_MAX_BATCH_SIZE = 32
ENCODER_MAX_SEQ_LEN = 1024

# function from https://github.com/jax-ml/jax/discussions/20538#discussioncomment-11156356
def grid_sample(
        image: jnp.ndarray,
        coords: jnp.ndarray,
        mode: str = 'linear',
        index='ij',
) -> jnp.ndarray:
    """
    Sample an image at arbitrary coordinates.

    Args:
        image: Array of shape [B, H, W, C] or [B, H, W, D, C]
        coords: Array of shape [B, h, w, 2] or [B, h, w, d, 3] containing coordinates in [-1, 1] range
        mode: Interpolation mode ('linear'/'bilinear'/'trilinear' or 'nearest')
    Returns:
        Interpolated values of shape [B, h, w, C] or [B, h, w, d, C]
    """

    B, *spatial_dims, C = image.shape
    if index == 'xy':  # Careful about how coordinates are swapped in 3D array
        coords = jnp.concatenate((coords[..., 1:2], coords[..., 0:1], coords[..., 2:]), -1)
    elif index != 'ij':
        raise ValueError(f'Unsuported indexing type: {index}')

    # Scale coordinates from [-1, 1] to [0, H/W]
    coords = (coords + 1) * (jnp.array(spatial_dims) - 1) / 2
    b_idx = jnp.arange(0, B).reshape(B, *[1]*len(spatial_dims))

    if mode in {'linear', 'bilinear', 'trilinear'}:
        # Get corner coordinates
        i0 = jnp.floor(coords[..., 0]).astype(jnp.int32)
        j0 = jnp.floor(coords[..., 1]).astype(jnp.int32)
        i1 = i0 + 1
        j1 = j0 + 1
        # Clip coordinates to valid range
        i0 = jnp.clip(i0, 0, spatial_dims[0] - 1)
        i1 = jnp.clip(i1, 0, spatial_dims[0] - 1)
        j0 = jnp.clip(j0, 0, spatial_dims[1] - 1)
        j1 = jnp.clip(j1, 0, spatial_dims[1] - 1)

        # Calculate interpolation weights
        wi = coords[..., 0] - i0
        wi = wi[..., None]
        wj = coords[..., 1] - j0
        wj = wj[..., None]
        if len(spatial_dims) == 2:
            output = (
                    image[b_idx, i0, j0] * (1-wi) * (1-wj) +
                    image[b_idx, i1, j0] * wi * (1 - wj) +
                    image[b_idx, i0, j1] * (1 - wi) * wj +
                    image[b_idx, i1, j1] * wi * wj
            )
        else:
            k0 = jnp.floor(coords[..., 2]).astype(jnp.int32)
            k1 = k0 + 1
            k0 = jnp.clip(k0, 0, spatial_dims[2] - 1)
            k1 = jnp.clip(k1, 0, spatial_dims[2] - 1)
            wk = coords[..., 2] - k0
            wk = wk[..., None]
            output = (
                    image[b_idx, i0, j0, k0] * (1 - wi) * (1 - wj) * (1 - wk) +
                    image[b_idx, i1, j0, k0] * wi * (1 - wj) * (1 - wk) +
                    image[b_idx, i0, j1, k0] * (1 - wi) * wj * (1 - wk) +
                    image[b_idx, i0, j0, k1] * (1 - wi) * (1 - wj) * wk +
                    image[b_idx, i1, j0, k1] * wi * (1 - wj) * wk +
                    image[b_idx, i0, j1, k1] * (1 - wi) * wj * wk +
                    image[b_idx, i1, j1, k0] * wi * wj * (1 - wk) +
                    image[b_idx, i1, j1, k1] * wi * wj * wk
            )
    elif mode == 'nearest':
        # Round coordinates to nearest integer
        y = jnp.clip(jnp.round(coords[..., 0]).astype(jnp.int32), 0, spatial_dims[0] - 1)
        x = jnp.clip(jnp.round(coords[..., 1]).astype(jnp.int32), 0, spatial_dims[1] - 1)
        if len(spatial_dims) == 2:
            output = image[b_idx, y, x]
        else:
            z = jnp.clip(jnp.round(coords[..., 2]).astype(jnp.int32), 0, spatial_dims[2] - 1)
            output = image[b_idx, y, x, z]
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    return output


class VisionEncoder(nnx.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.conv1 = ColumnParallelConv2dPatch(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = dim**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(dim))

        self.positional_embedding_vlm = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, dim)
        )

        self.ln_pre = LayerNorm(dim)
        self.ln_post = LayerNorm(dim)

        self.transformer = _Transformer(
            dim,
            layers,
            heads,
            ENCODER_MAX_BATCH_SIZE,
            ENCODER_MAX_SEQ_LEN,
            mlp_ratio,
            act_layer=nn.GELU,
        )

        # NOTE: hack for the fixed res
        image_h, image_w = self.image_size
        patch_h, patch_w = self.patch_size
        idx_h, idx_w = image_h // patch_h, image_w // patch_w
        img_idx = jnp.arange(image_h * image_w // (patch_h * patch_w), dtype=jnp.int32)
        img_idx = img_idx.reshape(idx_h * idx_w, 1)
        img_idx = jnp.concat([img_idx, img_idx[:1]], axis=0)
        img_idx[-1, -1] = PackingIndex.ID_CLS_TOKEN

        packed_img_idx = jnp.empty(
            (img_idx.shape[0], img_idx.shape[1]),
            # TODO: Determine what to do with this original arg: `PackingIndex.NUM_METADATA - 1,`
            dtype=jnp.int32,
        )
        packed_img_idx[:, :, PackingIndex.Y] = img_idx // idx_w
        packed_img_idx[:, :, PackingIndex.X] = img_idx % idx_w
        packed_img_idx[:, :, PackingIndex.HEIGHT].fill_(idx_h)
        packed_img_idx[:, :, PackingIndex.WIDTH].fill_(idx_w)
        packed_img_idx[:, :, PackingIndex.IDX] = img_idx
        packed_img_idx = packed_img_idx.reshape(1, -1, PackingIndex.NUM_METADATA - 1)
        self.packed_img_idx = packed_img_idx  # for positional embedding load hook

        # compute rope freqs
        rope_freq = self.get_rope_freqs(dim // heads // 2)
        freqs_x = self.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.X] + 1)
        freqs_y = self.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.Y] + 1)
        freqs = jnp.concat([freqs_x, freqs_y], axis=-1).float().contiguous()[..., ::2]
        # disable RoPE for padding and cls tokens
        freqs = freqs.masked_fill(packed_img_idx[:, :, PackingIndex.IDX, None] < 0, 0)
        # compute complex freqs
        self.freq_cis = torch.view_as_complex(jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], dim=-1))
        # xlf automatically broadcasts
        self.freq_cis = self.freq_cis.squeeze(0)
        self.n_heads = heads // fs_init.get_model_parallel_world_size()

        self._register_load_state_dict_pre_hook(self.load_hook)

    def get_rope_freqs(self, dim, theta=10000):
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        return freqs

    @torch.amp.autocast("cuda", enabled=False)
    def compute_rope_freqs(self, freqs, t):
        freqs = jnp.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        return freqs

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool = True,
        missing_keys: List[str] = None,
        unexpected_keys: List[str] = None,
        error_msgs: List[str] = None,
        return_state_dict: bool = False,
    ) -> None:
        orig_pos_embed = state_dict.get(prefix + "positional_embedding")
        if orig_pos_embed is not None and orig_pos_embed.shape[-2:] != self.positional_embedding_vlm.shape[-2:]:
            raise ValueError(
                f"Positional embedding shape {orig_pos_embed.shape} does not match expected shape {self.positional_embedding_vlm.shape}"
            )

        batch_size, token_per_image, _ = self.packed_img_idx.shape
        # Input points for idx are [x, y, w, h]
        idx = self.packed_img_idx.reshape(batch_size * token_per_image, 1, -1)
        total_windows, window_size, _ = idx.shape

        # Grid values are [-1, 1] and coords are w, h
        grid = (
            (idx[:, :, [PackingIndex.X, PackingIndex.Y]] / idx[:, :, [PackingIndex.WIDTH, PackingIndex.HEIGHT]]) * 2 - 1
        )[None, ...]

        # In this mode, cls token has no position embedding
        if orig_pos_embed is not None:
            posemb = (
                orig_pos_embed[1:].view(1, self.grid_size[0], self.grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
            )
            posemb = posemb.to(device=grid.device, dtype=grid.dtype)
            sample = grid_sample(
                posemb, grid, padding_mode="zeros"
            )  # padding tokens / class token will get zero for posemb
            sample = sample.view(-1, total_windows, window_size).permute(1, 2, 0).contiguous()
            sample = jnp.where(
                idx[:, :, PackingIndex.IDX, None] == PackingIndex.ID_CLS_TOKEN,
                orig_pos_embed[0].view(1, 1, -1).to(device=sample.device, dtype=sample.dtype),
                sample,
            )

            new_pos_embed = sample.reshape(batch_size, token_per_image, -1)

            state_dict[prefix + "positional_embedding_vlm"] = new_pos_embed.squeeze(0)

        if return_state_dict:
            return state_dict

    def apply_class_embedding(self, x):
        x = jnp.concat(
            [
                x,
                self.class_embedding.to(x.dtype)
                + jnp.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        return x

    def __call__(self, images: Array | KVTensor) -> Array | KVTensor:
        # NOTE: in Llama4 bsz=bsz*num_tiles, num_chunks=1
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, h, w = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, h, w = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, h, w)
        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, h, w)
        x = self.conv1(x)  # shape = [*, width, grid ** 2]
        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        if self.positional_embedding_vlm is not None:
            x = x + self.positional_embedding_vlm.to(x.dtype)

        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        x = self.ln_pre(x)
        x = x.view(bsz * num_concurrent_media, -1, dim)
        freq_cis = self.freq_cis.to(images.device)

        tf_output = self.transformer(
            x,
            freq_cis=freq_cis,
        )

        int_x = None
        if isinstance(tf_output, tuple):
            x, int_x = tf_output
        else:
            x = tf_output
        x = self.ln_post(x)

        # remove cls token output
        x = x[:, :-1, :]

        # add and output x + int_x features
        if int_x is not None:
            int_x = int_x[:, :-1, :, :]
            int_x = int_x.reshape(bsz * num_concurrent_media, ntok - 1, -1)
            x = jnp.concat([x, int_x], axis=-1)

        return x
