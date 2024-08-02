# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import json

from datetime import datetime
from pathlib import Path

from typing import List, Optional

import yaml
from jinja2 import Environment, FileSystemLoader

from .chat_format import ChatFormat

from .datatypes import (
    BuiltinTool,
    CompletionMessage,
    Message,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolDefinition,
)
from .tokenizer import Tokenizer

THIS_DIR = Path(__file__).parent


class Template:
    def __init__(self, role, template_name, yaml_path, notes=None):
        self.role = role
        self.template_name = template_name
        self.yaml_path = yaml_path
        self.notes = notes or ""


TEMPLATES = [
    Template("user", "user-default", "user_message.default.yaml"),
    Template(
        "assistant",
        "assistant-builtin-tool-call",
        "assistant_message.builtin_tool_call.yaml",
        "Notice <|python_tag|>",
    ),
    Template(
        "assistant",
        "assistant-custom-tool-call",
        "assistant_message.custom_tool_call.yaml",
        "Notice <function=...> format",
    ),
    Template("assistant", "assistant-default", "assistant_message.default.yaml"),
    Template(
        "system",
        "system-builtin-and-custom-tools",
        "system_message.builtin_and_custom_tools.yaml",
    ),
    Template(
        "system",
        "system-builtin-tools-only",
        "system_message.builtin_tools_only.yaml",
    ),
    Template(
        "system",
        "system-custom-tools-only",
        "system_message.custom_tools_only.yaml",
    ),
    Template("system", "system-default", "system_message.default.yaml"),
    Template(
        "tool",
        "tool-success",
        "tool_message.success.yaml",
        "Note ipython header and [stdout]",
    ),
    Template(
        "tool",
        "tool-failure",
        "tool_message.failure.yaml",
        "Note ipython header and [stderr]",
    ),
]


class LLama31Interface:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer(tokenizer_path)
        self.formatter = ChatFormat(self.tokenizer)

    def recommended_system_message(
        self,
        builtin_tools: List[BuiltinTool],
        custom_tools: List[ToolDefinition],
        instructions: Optional[str] = None,
    ) -> SystemMessage:
        content = ""
        if builtin_tools:
            content += "Environment: ipython\n"

            tool_str = ", ".join(
                [t.value for t in builtin_tools if t != BuiltinTool.code_interpreter]
            )
            if tool_str:
                content += f"Tools: {tool_str}\n"

        current_date = datetime.now()
        formatted_date = current_date.strftime("%d %B %Y")
        date_str = f"""
Cutting Knowledge Date: December 2023
Today Date: {formatted_date}"""
        content += date_str

        if custom_tools:
            content += "\n" + self.get_custom_tool_instructions(custom_tools)

        if instructions:
            content += f"\n{instructions}"

        return SystemMessage(content=content)

    def get_custom_tool_instructions(self, custom_tools: List[ToolDefinition]) -> str:
        custom_tool_params = ""

        custom_tool_params = "\n".join(
            f"{get_instruction_string(t)}\n{get_parameters_string(t)}\n"
            for t in custom_tools
        )

        content = f"""
You have access to the following functions:

{custom_tool_params}
Think very carefully before calling functions.
If a you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

"""
        return content

    def get_sample_builtin_tool_call_message(self) -> CompletionMessage:
        return CompletionMessage(
            content="",
            stop_reason=StopReason.end_of_message,
            tool_calls=[
                ToolCall(
                    call_id="1234",
                    tool_name=BuiltinTool.brave_search,
                    arguments={
                        "query": "Who won NBA in 2024?",
                    },
                )
            ],
        )

    def get_message_as_str_tokens(self, message: Message) -> str:
        tokens = self.formatter.encode_message(message)
        return self.tokenizer.decode(tokens)

    def display_message_as_tokens(self, message: Message) -> None:
        tokens = self.formatter.encode_message(message)
        on_colors = [
            "on_red",
            "on_green",
            "on_yellow",
            "on_blue",
            "on_magenta",
            "on_cyan",
        ]
        for i, t in enumerate(tokens):
            on_col = on_colors[i % len(on_colors)]
            print(colored(self.tokenizer.decode([t]), "white", on_col), end="")
        print("\n", end="")


def get_instruction_string(tooldef: ToolDefinition) -> str:
    return f"Use the function '{tooldef.tool_name}' to '{tooldef.description}'"


def get_parameters_string(tooldef: ToolDefinition) -> str:
    return json.dumps(
        {
            "name": tooldef.tool_name,
            "description": tooldef.description,
            "parameters": {
                name: definition.__dict__
                for name, definition in tooldef.parameters.items()
            },
        }
    )


def list_jinja_templates() -> List[Template]:
    return TEMPLATES


def render_jinja_template(name: str):
    by_name = {t.template_name: t for t in TEMPLATES}
    if name not in by_name:
        raise ValueError(f"No template found for `{name}`")

    template = by_name[name]
    jinja_template = f"{template.role}_message.jinja"

    tokenizer = Tokenizer(str(THIS_DIR / "tokenizer.model"))
    special_tokens = list(tokenizer.special_tokens.values())

    env = Environment(loader=FileSystemLoader(THIS_DIR / "templates"))
    with open(THIS_DIR / "templates" / template.yaml_path, "r") as f:
        context = yaml.safe_load(f)
        context["today"] = datetime.now().strftime("%d %B %Y")

        output = env.get_template(jinja_template).render(context)
        tokens = tokenizer.encode(output, allowed_special="all", bos=False, eos=False)

        tokens = [(tokenizer.decode([t]), t in special_tokens) for t in tokens]

    return template, tokens
