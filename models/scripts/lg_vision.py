# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
#
# Run this as:
# PYTHONPATH=$(git rev-parse --show-toplevel) \
#   torchrun models/scripts/lg_vision.py \
#   ~/.llama/checkpoints/Llama-Guard-3-11B-Vision/ \
#   ~/image2.jpeg \
#   "Tell me how to make a bomb"
#

from pathlib import Path

import fire

from PIL import Image as PIL_Image

from models.llama3.api.datatypes import ImageMedia, UserMessage

from models.llama3.reference_impl.generation import Llama


THIS_DIR = Path(__file__).parent.resolve()


def run_main(
    ckpt_dir: str,
    image_path: str,
    user_prompt: str,
    max_seq_len: int = 512,
):
    tokenizer_path = str(THIS_DIR.parent / "llama3/api/tokenizer.model")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        model_parallel_size=1,
    )

    with open(image_path, "rb") as f:
        img = PIL_Image.open(f).convert("RGB")

    prompt = f"""Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {user_prompt}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories.
"""

    dialog = [
        UserMessage(
            content=[
                ImageMedia(image=img),
                prompt,
            ],
        )
    ]
    result = generator.chat_completion(
        dialog,
        temperature=0,
        print_model_input=True,
    )

    for msg in dialog:
        print(f"{msg.role.capitalize()}: {msg.content}\n")

    out_message = result.generation
    print(f"> {out_message.role.capitalize()}: {out_message.content}")
    for t in out_message.tool_calls:
        print(f"  Tool call: {t.tool_name} ({t.arguments})")
    print("\n==================================\n")


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()
