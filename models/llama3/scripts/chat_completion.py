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

from models.datatypes import RawMediaItem, RawMessage, RawTextItem, StopReason
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

    dialogs = [
        [RawMessage(role="user", content="what is the recipe of mayonnaise?")],
        [
            RawMessage(
                role="user",
                content="I am going to Paris, what should I see?",
            ),
            RawMessage(
                role="assistant",
                content="""\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
                stop_reason=StopReason.end_of_turn,
            ),
            RawMessage(role="user", content="What is so great about #1?"),
        ],
        [
            RawMessage(role="system", content="Always answer with Haiku"),
            RawMessage(role="user", content="I am going to Paris, what should I see?"),
        ],
        [
            RawMessage(role="system", content="Always answer with emojis"),
            RawMessage(role="user", content="How to go from Beijing to NY?"),
        ],
    ]
    if generator.args.vision_chunk_size > 0:
        with open(THIS_DIR / "../../resources/dog.jpg", "rb") as f:
            img = f.read()

        dialogs.append(
            [
                RawMessage(
                    role="user",
                    content=[
                        RawMediaItem(data=BytesIO(img)),
                        RawTextItem(text="Describe this image in two sentences"),
                    ],
                ),
            ]
        )

    for dialog in dialogs:
        for msg in dialog:
            print(f"{msg.role.capitalize()}: {msg.content}\n")

        batch = [dialog]
        for token_results in generator.chat_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_seq_len,
        ):
            result = token_results[0]
            if result.finished:
                break

            cprint(result.text, color="yellow", end="")
        print("\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
