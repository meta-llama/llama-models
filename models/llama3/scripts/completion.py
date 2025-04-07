# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from io import BytesIO
from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

from models.datatypes import RawMediaItem
from models.llama3.generation import Llama3

import os
import torch


THIS_DIR = Path(__file__).parent


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
):
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
    )

    interleaved_contents = [
        "The color of the sky is blue but sometimes it can also be",
        """\
apple is pomme,
bannana is banane,
cherry is""",
        "1, 2, 3, 5, 8, 13",
        "ba ba black sheep, have you any wool?",
    ]
    if generator.args.vision_chunk_size > 0:
        with open(THIS_DIR / "../../resources/dog.jpg", "rb") as f:
            img = f.read()

        interleaved_contents.append(
            [
                RawMediaItem(type="image", data=BytesIO(img)),
                "If I had to write a haiku for this one",
            ]
        )

    for content in interleaved_contents:
        cprint(f"{content}", end="")
        batch = [content]
        for token_results in generator.completion(
            batch,
            temperature=temperature,
            top_p=top_p,
        ):
            result = token_results[0]
            if result.finished:
                break

            cprint(result.text, color="yellow", end="")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
