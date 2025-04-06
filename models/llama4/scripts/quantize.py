# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import json
import os
import sys
from pathlib import Path
from typing import Optional

import fire
import torch

from models.llama4.args import ModelArgs
from models.llama4.generation import QuantizationMode
from models.llama4.model import MoE, Transformer, TransformerBlock
from models.quantize_impls import int4_row_quantize, pack_int4

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

    print("Using efficient FP8/INT4 operators in FBGEMM.")
except ImportError:
    print("No efficient FP8/INT4 operators. Please install FBGEMM.")
    raise

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

# You can run this script with:
#
# torchrun --nproc-per-node=8 -m models.llama4.scripts.quantize \
#     --ckpt_dir /path/to/llama4/checkpoints \
#     --output_dir /path/to/output/dir \
#     --quantization_mode fp8_mixed \
#     --world_size 8


def ffn_quantize(
    ckpt_dir: str,
    output_dir: str,
    quantization_mode: str,
    world_size: Optional[int] = None,
) -> None:
    """
    Quantizes BF16 weights using "rowwise fp8" quantization or "int4-weight-bf16-activation (int4_mixed)" quantization. This method works well for FFNs.

    It produces two outputs:
     - quantized weights (in consolidated.0X.pth)
     - fp8 scales (in fp8_scales_X.pt)

    It produces three outputs for int4_mixed:
     - quantized weights (in consolidated.0X.pth)
     - int4 scales (in int4_scales_X.pt)

    The keys in the fp8/int4 scales are named so that they can be loaded using quantization/loader.py

    Args:
        ckpt_dir (str): The directory containing the checkpoint files.
        output_dir (str): The directory to save the quantized checkpoint and fp8/int4 scales.
        world_size (Optional[int]): The number of GPUs to use for model parallelization.
    """
    print(f"checkpoint_dir: {ckpt_dir} output_dir: {output_dir} world_size: {world_size}")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    if get_model_parallel_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

    seed = 1
    torch.manual_seed(seed)

    checkpoints = set(Path(ckpt_dir).glob("*.pth"))
    assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
    assert world_size == len(checkpoints), (
        f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    )
    checkpoints = sorted(checkpoints)
    ckpt_idx = get_model_parallel_rank()
    ckpt_path = checkpoints[ckpt_idx]

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(**params, max_seq_len=1024, max_batch_size=1)

    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model = Transformer(model_args)

    print(f"Quantizing {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        print("Reading checkpoint...")
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)
        print("Done...")

    print("Loading state dict...")
    model.load_state_dict(checkpoint, strict=False)
    print("Done...")

    fp8_scales = {}
    int4_scales = {}
    old_keys = set(checkpoint.keys())
    new_state_dict = checkpoint
    for _, block in model.named_modules():
        if isinstance(block, TransformerBlock):
            if block.layer_id == 0 or block.layer_id == (model.n_layers - 1):
                continue

            if not isinstance(block.feed_forward, MoE):
                continue

            # IMPORTANT NOTE:
            #  (1) Keys for weights are exactly as in the original state dict
            #  (2) Keys for fp8/int4 scales are according to re-mapped keys based on the load_hooks
            #
            # Why is it? In any case, things are confusing. However, this limits the confusion in the loading
            # code. The load_hooks remain simple, and in the fp8/int4 loading code at runtime, we don't need to
            # do any key re-mapping.
            #
            # Secondly:
            #   (1) MoE experts weights are transposed back so once again the loading code remains same for
            #       on-the-fly vs. from-pre-quantized weights
            #   (2) Scales however are _not_ transposed back

            prefix = f"layers.{block.layer_id}.feed_forward"
            moe = block.feed_forward

            if quantization_mode == QuantizationMode.int4_mixed:
                for key in ("w1", "w3", "w2"):
                    param = getattr(moe.experts, key)
                    shape = param.shape
                    weight = param.transpose(1, 2).contiguous()

                    if weight.ndim >= 3:
                        wq, scale = zip(*[int4_row_quantize(i.cuda()) for i in weight])
                        wq = torch.stack([pack_int4(i.cuda()) for i in wq], dim=0)
                        w_scale = torch.stack(scale, dim=0)
                    else:
                        wq, w_scale = int4_row_quantize(weight.cuda())
                        wq = pack_int4(wq.cuda())

                    state_dict_key_map = {
                        "w1": "moe_w_in_eD_F",
                        "w2": "moe_w_out_eF_D",
                        "w3": "moe_w_swiglu_eD_F",
                    }
                    # we must save this as 2D in the state dict, since loading code expects 2D weights
                    new_shape = (-1, shape[-1])
                    wq = wq.transpose(1, 2).reshape(*new_shape).contiguous()

                    # torch.nn.Parameter requires weights be floating point, so we cast packed int8 (2 * int4) to float8_e4m3fn, and will cast back to int8 in loading code
                    new_state_dict[f"{prefix}.experts.{state_dict_key_map[key]}"] = torch.nn.Parameter(
                        wq.view(torch.float8_e4m3fn)
                    )
                    int4_scales[f"{prefix}.experts.{key}"] = w_scale
                    print(f"Quantized {prefix}.experts.{state_dict_key_map[key]} {wq.shape=} {w_scale.shape=}")

            else:
                for key in ("w1", "w3", "w2"):
                    param = getattr(moe.experts, key)
                    shape = param.shape
                    wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(param.transpose(1, 2).contiguous())

                    state_dict_key_map = {
                        "w1": "moe_w_in_eD_F",
                        "w2": "moe_w_out_eF_D",
                        "w3": "moe_w_swiglu_eD_F",
                    }
                    # we must save this as 2D in the state dict, since loading code expects 2D weights
                    new_shape = (-1, shape[-1])
                    wq = wq.transpose(1, 2).reshape(*new_shape).contiguous()

                    new_state_dict[f"{prefix}.experts.{state_dict_key_map[key]}"] = torch.nn.Parameter(wq)
                    fp8_scales[f"{prefix}.experts.{key}"] = w_scale
                    print(f"Quantized {prefix}.experts.{state_dict_key_map[key]} {wq.shape=} {w_scale.shape=}")

    new_keys = set(new_state_dict.keys())
    assert old_keys == new_keys, f"old_keys != new_keys: {old_keys - new_keys}"

    if quantization_mode == QuantizationMode.int4_mixed:
        print("Saving int4 scales")
        int4_scales_path = os.path.join(output_dir, f"int4_scales_{ckpt_idx}.pt")
        torch.save(int4_scales, int4_scales_path)
    else:
        print("Saving fp8 scales")
        fp8_scales_path = os.path.join(output_dir, f"fp8_scales_{ckpt_idx}.pt")
        torch.save(fp8_scales, fp8_scales_path)

    ckpt_path = os.path.join(
        output_dir,
        "consolidated.{:02d}.pth".format(ckpt_idx),
    )
    print(f"Saving checkpoint to {ckpt_path}")
    torch.save(new_state_dict, ckpt_path)

    torch.distributed.barrier()


def main():
    fire.Fire(ffn_quantize)


if __name__ == "__main__":
    main()
