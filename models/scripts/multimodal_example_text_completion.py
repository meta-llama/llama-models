# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import Optional

import fire

from llama_models.llama3.api.datatypes import ImageMedia

from llama_models.llama3.reference_impl.generation import Llama

from PIL import Image as PIL_Image
from termcolor import cprint


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    model_parallel_size: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )

    with open(THIS_DIR / "resources/dog.jpg", "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    with open(THIS_DIR / "resources/pasta.jpeg", "rb") as f:
        img2 = PIL_Image.open(f).convert("RGB")

    interleaved_contents = [
        # text only
        "The color of the sky is blue but sometimes it can also be",
        # image understanding
        [
            ImageMedia(image=img),
            "If I had to write a haiku for this one",
        ],
    ]

    for content in interleaved_contents:
        result = generator.text_completion(
            content,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        cprint(f"{content}", end="")
        cprint(f"{result.generation}", color="yellow")
        print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
