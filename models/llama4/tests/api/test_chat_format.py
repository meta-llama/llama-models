# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import json
import unittest
from unittest.mock import Mock

from llama_models.datatypes import BuiltinTool, StopReason
from llama_models.llama4.chat_format import ChatFormat
from llama_models.llama4.tokenizer import Tokenizer


class TestChatFormatArgumentsJson(unittest.TestCase):
    """Test that ToolCall objects include the arguments_json parameter."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the tokenizer to avoid dependency issues in tests
        self.mock_tokenizer = Mock(spec=Tokenizer)
        self.chat_format = ChatFormat(self.mock_tokenizer)

    def test_arguments_json_included_in_custom_tool_call(self):
        """Test that arguments_json is included for custom tool calls."""
        # Mock content that represents a custom tool call in python_list format
        content = '[get_weather(location="San Francisco", units="metric")]'

        # Mock the ToolUtils.maybe_extract_custom_tool_call to return our test data
        with unittest.mock.patch(
            "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_custom_tool_call"
        ) as mock_extract:
            mock_extract.return_value = ("get_weather", {"location": "San Francisco", "units": "metric"})

            # Call the method under test
            result = self.chat_format.decode_assistant_message_from_content(content, StopReason.end_of_turn)

            # Verify the result
            self.assertEqual(len(result.tool_calls), 1)
            tool_call = result.tool_calls[0]

            # Verify all expected fields are present
            self.assertIsNotNone(tool_call.call_id)
            self.assertEqual(tool_call.tool_name, "get_weather")
            self.assertEqual(tool_call.arguments, {"location": "San Francisco", "units": "metric"})

            # This is the critical test - verify arguments_json is included
            self.assertIsNotNone(tool_call.arguments_json)
            self.assertEqual(tool_call.arguments_json, '{"location": "San Francisco", "units": "metric"}')

            # Verify arguments_json is valid JSON that matches arguments
            parsed_args = json.loads(tool_call.arguments_json)
            self.assertEqual(parsed_args, tool_call.arguments)

    def test_arguments_json_included_in_builtin_tool_call(self):
        """Test that arguments_json is included for builtin tool calls."""
        # Mock content that represents a builtin tool call
        content = "search for weather in San Francisco"

        # Mock the ToolUtils methods to simulate builtin tool extraction
        with (
            unittest.mock.patch(
                "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_custom_tool_call"
            ) as mock_custom,
            unittest.mock.patch(
                "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_builtin_tool_call"
            ) as mock_builtin,
        ):
            mock_custom.return_value = None  # No custom tool
            mock_builtin.return_value = ("brave_search", "weather in San Francisco")

            # Call the method under test
            result = self.chat_format.decode_assistant_message_from_content(content, StopReason.end_of_turn)

            # Verify the result
            self.assertEqual(len(result.tool_calls), 1)
            tool_call = result.tool_calls[0]

            # Verify all expected fields are present
            self.assertIsNotNone(tool_call.call_id)
            self.assertEqual(tool_call.tool_name, BuiltinTool.brave_search)
            self.assertEqual(tool_call.arguments, {"query": "weather in San Francisco"})

            # This is the critical test - verify arguments_json is included
            self.assertIsNotNone(tool_call.arguments_json)
            self.assertEqual(tool_call.arguments_json, '{"query": "weather in San Francisco"}')

            # Verify arguments_json is valid JSON that matches arguments
            parsed_args = json.loads(tool_call.arguments_json)
            self.assertEqual(parsed_args, tool_call.arguments)

    def test_arguments_json_included_in_code_interpreter_call(self):
        """Test that arguments_json is included for code interpreter tool calls."""
        # Mock content that represents a code interpreter tool call
        content = "run some python code"

        # Mock the ToolUtils methods to simulate code interpreter tool extraction
        with (
            unittest.mock.patch(
                "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_custom_tool_call"
            ) as mock_custom,
            unittest.mock.patch(
                "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_builtin_tool_call"
            ) as mock_builtin,
        ):
            mock_custom.return_value = None  # No custom tool
            mock_builtin.return_value = ("code_interpreter", "print('Hello, World!')")

            # Call the method under test
            result = self.chat_format.decode_assistant_message_from_content(content, StopReason.end_of_turn)

            # Verify the result
            self.assertEqual(len(result.tool_calls), 1)
            tool_call = result.tool_calls[0]

            # Verify all expected fields are present
            self.assertIsNotNone(tool_call.call_id)
            self.assertEqual(tool_call.tool_name, BuiltinTool.code_interpreter)
            self.assertEqual(tool_call.arguments, {"query": "print('Hello, World!')"})

            # This is the critical test - verify arguments_json is included
            self.assertIsNotNone(tool_call.arguments_json)
            self.assertEqual(tool_call.arguments_json, '{"query": "print(\'Hello, World!\')"}')

            # Verify arguments_json is valid JSON that matches arguments
            parsed_args = json.loads(tool_call.arguments_json)
            self.assertEqual(parsed_args, tool_call.arguments)

    def test_arguments_json_with_complex_arguments(self):
        """Test that arguments_json handles complex arguments correctly."""
        # Mock content that represents a custom tool call with complex but supported arguments
        content = '[complex_function(title="My Title", numbers=[1, 2, 3], enabled=true, score=3.14)]'

        # Mock the ToolUtils.maybe_extract_custom_tool_call to return our test data
        with unittest.mock.patch(
            "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_custom_tool_call"
        ) as mock_extract:
            complex_args = {"title": "My Title", "numbers": [1, 2, 3], "enabled": True, "score": 3.14}
            mock_extract.return_value = ("complex_function", complex_args)

            # Call the method under test
            result = self.chat_format.decode_assistant_message_from_content(content, StopReason.end_of_turn)

            # Verify the result
            self.assertEqual(len(result.tool_calls), 1)
            tool_call = result.tool_calls[0]

            # Verify all expected fields are present
            self.assertIsNotNone(tool_call.call_id)
            self.assertEqual(tool_call.tool_name, "complex_function")
            self.assertEqual(tool_call.arguments, complex_args)

            # This is the critical test - verify arguments_json is included
            self.assertIsNotNone(tool_call.arguments_json)

            # Verify arguments_json is valid JSON that matches arguments
            parsed_args = json.loads(tool_call.arguments_json)
            self.assertEqual(parsed_args, tool_call.arguments)

            # Verify the JSON string contains the expected structure
            expected_json = '{"title": "My Title", "numbers": [1, 2, 3], "enabled": true, "score": 3.14}'
            self.assertEqual(tool_call.arguments_json, expected_json)

    def test_no_tool_calls_when_no_tools_detected(self):
        """Test that no tool calls are created when no tools are detected."""
        content = "Just a regular response with no tool calls."

        # Mock the ToolUtils methods to return None (no tools detected)
        with (
            unittest.mock.patch(
                "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_custom_tool_call"
            ) as mock_custom,
            unittest.mock.patch(
                "llama_models.llama3.tool_utils.ToolUtils.maybe_extract_builtin_tool_call"
            ) as mock_builtin,
        ):
            mock_custom.return_value = None
            mock_builtin.return_value = None

            # Call the method under test
            result = self.chat_format.decode_assistant_message_from_content(content, StopReason.end_of_turn)

            # Verify no tool calls are created
            self.assertEqual(len(result.tool_calls), 0)
            self.assertEqual(result.content, content)


if __name__ == "__main__":
    unittest.main()
