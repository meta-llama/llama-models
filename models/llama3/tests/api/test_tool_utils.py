# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from llama_models.llama3.api.tool_utils import ToolUtils


class TestToolUtils:

    def test_maybe_extract_custom_tool_call(self):
        single_tool_call = (
            """<function=getWeather>{"location":"New York","date":"2023-08-05"}"""
        )
        res = ToolUtils.maybe_extract_custom_tool_call(single_tool_call)
        tool_name, args = res
        assert tool_name == "getWeather"
        assert args == {"location": "New York", "date": "2023-08-05"}
