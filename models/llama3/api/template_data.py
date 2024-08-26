from ..prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    SystemDefaultGenerator,
    ToolResponseGenerator,
)
from .datatypes import ToolPromptFormat

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
        "tool_prompt_format": ToolPromptFormat.json,
    }


def system_message_custom_tools_only_function_tag_format():
    return {
        "builtin_tools": [],
        "custom_tools": JsonCustomToolGenerator().data_examples()[0],
        "instruction": INSTRUCTION,
        "tool_prompt_format": ToolPromptFormat.function_tag,
    }


def system_message_builtin_and_custom_tools():
    return {
        "builtin_tools": BuiltinToolGenerator().data_examples()[0],
        "custom_tools": JsonCustomToolGenerator().data_examples()[0],
        "instruction": INSTRUCTION,
        "tool_prompt_format": ToolPromptFormat.json,
    }


def tool_success():
    return ToolResponseGenerator().data_examples()[0]


def tool_failure():
    return ToolResponseGenerator().data_examples()[1]
