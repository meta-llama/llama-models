# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from ..prompt_templates import (
    BuiltinToolGenerator,
    JsonCustomToolGenerator,
    ToolResponseGenerator,
)

INSTRUCTION = "You are a helpful assistant."


def system_message_builtin_tools_only():
    return {
        "builtin_tools": BuiltinToolGenerator().data_examples()[0],
        "custom_tools": [],
        "instruction": INSTRUCTION,
    }


def system_message_custom_tools_only():
    return {
        "builtin_tools": [],
        "custom_tools": JsonCustomToolGenerator().data_examples()[0],
        "instruction": INSTRUCTION,
    }


def system_message_builtin_and_custom_tools():
    return {
        "builtin_tools": BuiltinToolGenerator().data_examples()[0],
        "custom_tools": JsonCustomToolGenerator().data_examples()[0],
        "instruction": INSTRUCTION,
    }


def tool_success():
    return ToolResponseGenerator().data_examples()[0]


def tool_failure():
    return ToolResponseGenerator().data_examples()[1]
