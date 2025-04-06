# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import codecs
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, List, Literal, Optional

import jax
import jax.numpy as jnp
from termcolor import cprint

from . import common_types
from .args import ModelArgs
from .chat_format import ChatFormat, RawContent, RawMessage
from .checkpoint import load_state_dict
from .common_types import Array, KVTensor
from .datatypes import LLMInput, MaskedEmbedding, TransformerInput
from .model import Transformer
from .tokenizer import Tokenizer


@dataclass
class GenerationResult:
    token: int
    text: str

    source: Literal["input", "output"]

    # index within the batch
    batch_idx: int
    # whether generation for this item is already finished. note that tokens can
    # get returned even afterwards since other items in the batch can still be generating tokens
    finished: bool
    # because a batch is parallel processed, useful decoding for one item can correspond to processing
    # pad tokens or tokens beyond EOS for other items. we could have decided to return None for this case
    # but it's more convenient to return a list of GenerationResult and filter out the ignored tokens
    ignore_token: bool

    logprobs: Optional[List[float]] = None


torch.serialization.add_safe_globals([io.BytesIO, codecs.encode])


class QuantizationMode(str, Enum):
    none = "none"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"


class Llama4:
    @staticmethod
    def build(
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        world_size: Optional[int] = None,
        quantization_mode: Optional[str] = None,
        seed: int = 1,
    ):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not model_parallel_is_initialized():
            if world_size is None:
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(world_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()

        ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(ckpt_paths) > 0, f"no checkpoint files found in {ckpt_dir}"
        print(f"Loading a checkpoint (shards={len(ckpt_paths)}, current-mp-size={world_size})")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            **params,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        tokenizer = Tokenizer.get_instance()

        # TODO: params.json should always have correct vocab_size
        if model_args.vocab_size == -1:
            model_args.vocab_size = tokenizer.n_words
        assert model_args.vocab_size == tokenizer.n_words, f"{model_args.vocab_size=} vs. {tokenizer.n_words=} mismatch"
        print("Model args:\n", model_args.model_dump_json(indent=2))

        state_dict = load_state_dict(ckpt_paths, model_args)
        print("Loaded checkpoint")
        common_types.set_floatx("bfloat16")
        if quantization_mode == QuantizationMode.fp8_mixed or quantization_mode == QuantizationMode.int4_mixed:
            from .quantization.loader import convert_to_quantized_model

            model = Transformer(model_args)
            print("Loading state dict...")
            model.load_state_dict(state_dict, strict=False)
            print("Done...")
            model = convert_to_quantized_model(model, ckpt_dir, quantization_mode)
        else:
            model = Transformer(model_args)
            print("Loading state dict...")
            model.load_state_dict(state_dict, strict=False)
            print("Done...")
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama4(model, tokenizer, model_args)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer, vision_args=args.vision_args)

    @torch.inference_mode()
    def generate(
        self,
        llm_inputs: List[LLMInput],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        print_model_input: bool = False,
        logits_processor: Optional[Callable[[Array | KVTensor, Array | KVTensor], Array | KVTensor]] = None,
    ) -> Generator[List[GenerationResult], None, None]:
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.args.max_seq_len:
            max_gen_len = self.model.args.max_seq_len - 1

        params = self.model.args

        print_model_input = print_model_input or os.environ.get("LLAMA_MODELS_DEBUG", "0") == "1"
        if print_model_input and get_model_parallel_rank() == 0:
            cprint("Input to model:\n", "yellow")
            for inp in llm_inputs:
                tokens_to_print = [t for t in inp.tokens]
                cprint(self.tokenizer.decode(tokens_to_print), "grey")
        prompt_tokens = [inp.tokens for inp in llm_inputs]

        bsz = len(llm_inputs)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        if max_prompt_len >= params.max_seq_len:
            cprint(f"Out of token budget {max_prompt_len} vs {params.max_seq_len}", "red")
            return

        total_len = min(max_gen_len + max_prompt_len, params.max_seq_len)

        pad_id = self.tokenizer.pad_id
        tokens = jnp.full((bsz, total_len), pad_id, dtype=jnp.int64, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=jnp.int64, device="cuda")
        if logprobs:
            token_logprobs = jnp.zeros_like(tokens, dtype=common_types.floatx())

        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        if echo:
            for i in range(max_prompt_len):
                results = []
                for j, t in enumerate(tokens[:, i]):
                    results.append(
                        GenerationResult(
                            token=t.item(),
                            text=self.tokenizer.decode([t.item()]),
                            source="input",
                            logprobs=(token_logprobs[j, i : i + 1].tolist() if logprobs else None),
                            batch_idx=j,
                            finished=False,
                            ignore_token=t.item() == pad_id,
                        )
                    )
                yield results

        stop_tokens = torch.tensor(self.tokenizer.stop_tokens, device="cuda")

        prev_pos = 0
        for cur_pos in range(min_prompt_len, total_len):
            image_embedding = None
            if prev_pos == 0 and any(inp.images is not None and len(inp.images) > 0 for inp in llm_inputs):
                image_mask = tokens[:, prev_pos:cur_pos] == self.tokenizer.special_tokens["<|patch|>"]
                image_mask = image_mask.unsqueeze(-1)
                h = self.model.tok_embeddings(tokens[:, prev_pos:cur_pos])

                image_batch = [inp.images if inp.images is not None else [] for inp in llm_inputs]
                image_embedding = MaskedEmbedding(
                    embedding=self.model.vision_embeddings(image_batch, image_mask, h),
                    mask=image_mask,
                )

            xformer_input = TransformerInput(
                tokens=tokens[:, prev_pos:cur_pos],
                tokens_position=prev_pos,
                image_embedding=image_embedding,
            )
            xformer_output = self.model.__call__(xformer_input)
            logits = xformer_output.logits
            if logits_processor is not None:
                logits = logits_processor(tokens[:, :cur_pos], logits)

            if temperature > 0:
                probs = jax.nn.softmax(logits[:, -1] / temperature, axis=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = jax.lax.argmax(logits[:, -1], axis=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = jnp.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            target = tokens[:, prev_pos + 1 : cur_pos + 1]
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=target,
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (jax.numpy.isin(next_token, stop_tokens))

            results = []
            for idx, t in enumerate(next_token):
                results.append(
                    GenerationResult(
                        token=t.item(),
                        text=self.tokenizer.decode([t.item()]),
                        source="output",
                        logprobs=(token_logprobs[idx, cur_pos : cur_pos + 1].tolist() if logprobs else None),
                        batch_idx=idx,
                        finished=eos_reached[idx],
                        ignore_token=cur_pos < len(prompt_tokens[idx]),
                    )
                )
            yield results

            prev_pos = cur_pos
            if all(eos_reached):
                break

    def completion(
        self,
        contents: List[RawContent],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Generator[List[GenerationResult], None, None]:
        llm_inputs = [self.formatter.encode_contents(c) for c in contents]
        for result in self.generate(
            llm_inputs=llm_inputs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
            echo=echo,
        ):
            if all(r.finished for r in result):
                break
            yield result

    def chat_completion(
        self,
        messages_batch: List[List[RawMessage]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Generator[List[GenerationResult], None, None]:
        llm_inputs = [self.formatter.encode_dialog_prompt(messages) for messages in messages_batch]
        for result in self.generate(
            llm_inputs=llm_inputs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
            echo=echo,
        ):
            if all(r.finished for r in result):
                break
            yield result


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (Array | KVTensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        Array | KVTensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = jax.lax.rev(jax.lax.sort(probs, dimesion=-1), dimesion=-1)
    probs_sum = jax.lax.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = jax.random.multinomial(probs_sort, n=1)
    next_token = jax.lax.gather(probs_idx, -1, next_token)
    return next_token
