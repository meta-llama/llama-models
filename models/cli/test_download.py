# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from unittest.mock import patch

from llama_models.cli.llama import LlamaModelsCLIParser


def test_download_model_id_imports_safety_models():
    cli = LlamaModelsCLIParser()
    args = cli.parser.parse_args(
        ["download", "--model-id", "Prompt-Guard-86M", "--meta-url", "https://llamameta.net/mock"]
    )

    with patch("llama_models.cli.download._meta_download") as download:
        cli.run(args)

    download.assert_called_once()
    assert download.call_args.args[1] == "Prompt-Guard-86M"
