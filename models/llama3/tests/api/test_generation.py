# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import unittest

from pathlib import Path

import numpy as np
import pytest
from llama_models.llama3.api.datatypes import ImageMedia, SystemMessage, UserMessage

from llama_models.llama3.reference_impl.generation import Llama
from PIL import Image as PIL_Image

THIS_DIR = Path(__file__).parent


def build_generator(env_var: str):
    if env_var not in os.environ:
        raise ValueError(f"{env_var} must be specified for this test")

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    return Llama.build(
        ckpt_dir=os.environ[env_var],
        max_seq_len=128,
        max_batch_size=1,
        model_parallel_size=1,
    )


class TestTextModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = build_generator("TEXT_MODEL_CHECKPOINT_DIR")

    def test_run_generation(self):
        dialogs = [
            [
                SystemMessage(content="Always answer with Haiku"),
                UserMessage(content="I am going to Paris, what should I see?"),
            ],
            [
                SystemMessage(
                    content="Always answer with emojis",
                ),
                UserMessage(content="How to go from Beijing to NY?"),
            ],
        ]
        for dialog in dialogs:
            result = self.__class__.generator.chat_completion(
                dialog,
                temperature=0,
                logprobs=True,
            )

            out_message = result.generation
            self.assertTrue(len(out_message.content) > 0)
            shape = np.array(result.logprobs).shape
            # assert at least 10 tokens
            self.assertTrue(shape[0] > 10)
            self.assertEqual(shape[1], 1)


class TestVisionModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = build_generator("VISION_MODEL_CHECKPOINT_DIR")

    @unittest.skip("Disabling vision model test")
    @pytest.mark.skip(reason="Disabling vision model test")
    def test_run_generation(self):
        with open(
            THIS_DIR.parent.parent.parent / "scripts/resources/dog.jpg", "rb"
        ) as f:
            img = PIL_Image.open(f).convert("RGB")

        dialogs = [
            [
                UserMessage(
                    content=[
                        ImageMedia(image=img),
                        "Describe this image in two sentences",
                    ],
                )
            ],
            [
                UserMessage(
                    content="what is the recipe of mayonnaise in two sentences?"
                ),
            ],
        ]

        for dialog in dialogs:
            result = self.__class__.generator.chat_completion(
                dialog,
                temperature=0,
                logprobs=True,
            )

            out_message = result.generation
            self.assertTrue(len(out_message.content) > 0)
            shape = np.array(result.logprobs).shape
            # assert at least 10 tokens
            self.assertTrue(shape[0] > 10)
            self.assertEqual(shape[1], 1)
