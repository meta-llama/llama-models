# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os
import unittest
from pathlib import Path

import pytest
import torch

from llama_models.datatypes import RawMediaItem, RawMessage, RawTextItem
from llama_models.llama3.generation import Llama3

THIS_DIR = Path(__file__).parent


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]

    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return ""


def build_generator(env_var: str, device: str):
    if env_var not in os.environ:
        raise ValueError(f"{env_var} must be specified for this test")

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    return Llama3.build(ckpt_dir=os.environ[env_var], max_seq_len=128, max_batch_size=1, world_size=1, device=device)


class TestTextModelInference(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.generator = build_generator("TEXT_MODEL_CHECKPOINT_DIR", cls.device)

    def test_run_generation(self):
        dialogs = [
            [
                RawMessage(role="system", content="Always answer with Haiku"),
                RawMessage(role="user", content="I am going to Paris, what should I see?"),
            ],
            [
                RawMessage(role="system", content="Always answer with emojis"),
                RawMessage(role="user", content="How to go from Beijing to NY?"),
            ],
        ]
        for dialog in dialogs:
            batch = [dialog]
            out_message = []
            for token_results in self.__class__.generator.chat_completion(
                batch,
                temperature=0,
                logprobs=True,
            ):
                result = token_results[0]
                if result.finished:
                    break
                out_message += [result]
                self.assertTrue(len(result.text) > 0)
            # assert at least 10 tokens
            self.assertTrue(len(out_message) > 10)


@pytest.mark.skipif(get_device() == "", reason="No device available and none specified")
class TestTextModelInferenceOnDevice(TestTextModelInference):
    device = get_device()


class TestVisionModelInference(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.generator = build_generator("VISION_MODEL_CHECKPOINT_DIR", cls.device)

    @unittest.skip("Disabling vision model test")
    @pytest.mark.skip(reason="Disabling vision model test")
    def test_run_generation(self):
        with open(THIS_DIR.parent.parent.parent / "resources/dog.jpg", "rb") as f:
            img = f.read()

        dialogs = [
            [
                RawMessage(
                    role="user",
                    content=[
                        RawMediaItem(data=img),
                        RawTextItem(text="Describe this image in two sentences"),
                    ],
                )
            ],
            [
                RawMessage(
                    role="user",
                    content="what is the recipe of mayonnaise in two sentences?",
                )
            ],
        ]

        for dialog in dialogs:
            batch = [dialog]
            out_message = []
            for token_results in self.__class__.generator.chat_completion(
                batch,
                temperature=0,
                logprobs=True,
            ):
                result = token_results[0]
                if result.finished:
                    break
                out_message += [result]
                self.assertTrue(len(result.text) > 0)
            # assert at least 10 tokens
            self.assertTrue(len(out_message) > 10)


@pytest.mark.skipif(get_device() == "", reason="No device available and none specified")
class TestVisionModelInferenceOnDevice(TestVisionModelInference):
    device = get_device()
