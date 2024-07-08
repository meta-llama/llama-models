# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import glob
import json
import os
from datetime import datetime
from pathlib import Path

from typing import List, Optional

from termcolor import colored, cprint

from .chat_format import ChatFormat

from .datatypes import (
    BuiltinTool,
    CompletionMessage,
    Message,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolParamDefinition,
)
from .tokenizer import Tokenizer

THIS_DIR = Path(__file__).parent

TEMPLATES = {}
for role in ("system", "assistant", "tool", "user"):
    for path in glob.glob(str(THIS_DIR / "templates" / f"{role}_message*.yaml")):
        name = os.path.basename(path)
        name = name.replace("_", "-").replace(".yaml", "").replace(".", "-")
        TEMPLATES[name] = (role, path, f"{role}_message.jinja")


class LLama3_1_Interface:
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


def list_jinja_templates():
    global TEMPLATES

    for name in TEMPLATES.keys():
        print(f"{name}")


def render_jinja_template(name: str):
    if name not in TEMPLATES:
        raise ValueError(f"No template found for `{name}`")

    import yaml
    from jinja2 import Environment, FileSystemLoader

    tokenizer = Tokenizer(str(THIS_DIR / "tokenizer.model"))
    special_tokens = list(tokenizer.special_tokens.values())

    role, input_path, jinja_template = TEMPLATES[name]

    env = Environment(loader=FileSystemLoader(THIS_DIR / "templates"))
    template = env.get_template(jinja_template)

    with open(input_path, "r") as f:
        context = yaml.safe_load(f)
        context["today"] = datetime.now().strftime("%d %B %Y")

        output = template.render(context)
        tokens = tokenizer.encode(output, allowed_special="all", bos=False, eos=False)

        for t in tokens:
            decoded = tokenizer.decode([t])
            if t in special_tokens:
                cprint(decoded, "yellow", end="")
            else:
                print(decoded, end="")

        print("")
