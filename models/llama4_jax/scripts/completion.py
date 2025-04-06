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
from models.llama4.generation import Llama4

THIS_DIR = Path(__file__).parent


def run_main(
    checkpoint_dir: str,
    world_size: int = 1,
    max_seq_len: Optional[int] = 1024,
    max_batch_size: Optional[int] = 1,
    temperature: float = 0.6,
    top_p: float = 0.9,
    quantization_mode: Optional[str] = None,
):
    generator = Llama4.build(
        checkpoint_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
    )

    with open(THIS_DIR / "../../resources/dog.jpg", "rb") as f:
        img = f.read()

    interleaved_contents = [
        # text only
        "The color of the sky is blue but sometimes it can also be",
        "The capital of France is",
        # image understanding
        [
            RawMediaItem(type="image", data=BytesIO(img)),
            "If I had to write a haiku for this one",
        ],
    ]

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
