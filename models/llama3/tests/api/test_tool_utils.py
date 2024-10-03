# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.
import unittest

from llama_models.llama3.api.tool_utils import (
    is_valid_python_list,
    parse_python_list_for_function_calls,
    ToolUtils,
)


class TestToolUtils(unittest.TestCase):

    def test_maybe_extract_custom_tool_call(self):
        single_tool_call = (
            """<function=getWeather>{"location":"New York","date":"2023-08-05"}"""
        )
        res = ToolUtils.maybe_extract_custom_tool_call(single_tool_call)
        tool_name, args = res
        assert tool_name == "getWeather"
        assert args == {"location": "New York", "date": "2023-08-05"}


class TestPythonListCheck(unittest.TestCase):

    def test_valid_list_with_single_function_call(self):
        input_string = '[get_boiling_point(liquid_name="water", celcius=True)]'
        assert is_valid_python_list(input_string) is True

    def test_valid_list_with_multiple_function_calls(self):
        input_string = '[get_boiling_point(liquid_name="water", celcius=True), get_melting_point(substance="ice", kelvin=False)]'
        assert is_valid_python_list(input_string) is True

    def test_invalid_empty_list(self):
        input_string = "[]"
        assert is_valid_python_list(input_string) is False

    def test_invalid_list_with_non_function_call(self):
        input_string = '[get_boiling_point(liquid_name="water"), 42]'
        assert is_valid_python_list(input_string) is False

    def test_invalid_list_with_positional_args(self):
        input_string = '[get_boiling_point("water", True)]'
        assert is_valid_python_list(input_string) is False

    def test_invalid_nested_list(self):
        input_string = '[[get_boiling_point(liquid_name="water")]]'
        assert is_valid_python_list(input_string) is False

    def test_invalid_dict(self):
        input_string = '{"func": get_boiling_point(liquid_name="water")}'
        assert is_valid_python_list(input_string) is False

    def test_invalid_syntax(self):
        input_string = '[get_boiling_point(liquid_name="water", celcius=True'  # Missing closing bracket
        assert is_valid_python_list(input_string) is False

    def test_valid_list_with_boolean_args(self):
        input_string = (
            '[get_boiling_point(liquid_name="water", celcius=True, precise=False)]'
        )
        assert is_valid_python_list(input_string) is True

    def test_valid_list_with_numeric_args(self):
        input_string = "[calculate_volume(radius=5.2, height=10)]"
        assert is_valid_python_list(input_string) is True

    def test_invalid_bare_function_call(self):
        input_string = 'get_boiling_point(liquid_name="water")'
        assert is_valid_python_list(input_string) is False

    def test_invalid_extra_char_function_call(self):
        input_string = '[(get_boiling_point(liquid_name="water"),)]'
        assert is_valid_python_list(input_string) is False


class TestParsePythonList(unittest.TestCase):

    def test_single_function_call(self):
        input_string = '[get_boiling_point(liquid_name="water", celcius=True)]'
        expected = [("get_boiling_point", {"liquid_name": "water", "celcius": True})]
        assert parse_python_list_for_function_calls(input_string) == expected

    def test_multiple_function_calls(self):
        input_string = '[get_boiling_point(liquid_name="water", celcius=True), get_melting_point(substance="ice", kelvin=False)]'
        expected = [
            ("get_boiling_point", {"liquid_name": "water", "celcius": True}),
            ("get_melting_point", {"substance": "ice", "kelvin": False}),
        ]
        assert parse_python_list_for_function_calls(input_string) == expected

    def test_function_call_with_numeric_args(self):
        input_string = "[calculate_volume(radius=5.2, height=10)]"
        expected = [("calculate_volume", {"radius": 5.2, "height": 10})]
        assert parse_python_list_for_function_calls(input_string) == expected

    def test_function_call_with_mixed_type_args(self):
        input_string = (
            '[process_data(name="sample", value=42, active=True, ratio=3.14)]'
        )
        expected = [
            (
                "process_data",
                {"name": "sample", "value": 42, "active": True, "ratio": 3.14},
            )
        ]
        assert parse_python_list_for_function_calls(input_string) == expected

    def test_function_call_with_empty_args(self):
        input_string = "[initialize()]"
        expected = [("initialize", {})]
        assert parse_python_list_for_function_calls(input_string) == expected

    def test_function_call_with_string_containing_spaces(self):
        input_string = '[analyze_text(content="Hello, world!", language="en")]'
        expected = [("analyze_text", {"content": "Hello, world!", "language": "en"})]
        assert parse_python_list_for_function_calls(input_string) == expected

    def test_function_names_with_underscores_lists_and_dicts(self):
        input_string = '[get_boiling_point(liquid={"name":"water", "color":"translucent"}), convert_temperature(value=[100, 101], from_unit="C", to_unit="F")]'
        expected = [
            (
                "get_boiling_point",
                {"liquid": {"name": "water", "color": "translucent"}},
            ),
            (
                "convert_temperature",
                {"value": [100, 101], "from_unit": "C", "to_unit": "F"},
            ),
        ]
        assert parse_python_list_for_function_calls(input_string) == expected
