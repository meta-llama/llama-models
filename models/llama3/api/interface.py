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
from termcolor import colored

from ..prompt_templates import (
    BuiltinToolGenerator,
    FunctionTagCustomToolGenerator,
    JsonCustomToolGenerator,
    SystemDefaultGenerator,
    ToolResponseGenerator,
)

from . import template_data

from .chat_format import ChatFormat

from .datatypes import (
    BuiltinTool,
    CompletionMessage,
    Message,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolPromptFormat,
    ToolResponseMessage,
    UserMessage,
)
from .tokenizer import Tokenizer


THIS_DIR = Path(__file__).parent


class Template:
    def __init__(
        self,
        role,
        template_name,
        yaml_path=None,
        data_provider=None,
        notes=None,
    ):
        self.role = role
        self.template_name = template_name
        self.yaml_path = yaml_path
        self.data_provider = data_provider or ""
        self._notes = notes or ""

    @property
    def notes(self):
        default = "↵ represents newline"
        notes = default
        if self._notes:
            notes += "\n"
            notes += self._notes
        return notes


TEMPLATES = [
    Template(
        "user",
        "user-default",
        "user_message.default.yaml",
        "user_default",
    ),
    Template(
        "assistant",
        "assistant-builtin-tool-call",
        "assistant_message.builtin_tool_call.yaml",
        "assistant_builtin_tool_call",
        "Notice <|python_tag|>",
    ),
    Template(
        "assistant",
        "assistant-custom-tool-call",
        "assistant_message.custom_tool_call.yaml",
        "assistant_custom_tool_call",
        "Notice <function=...> format",
    ),
    Template(
        "assistant",
        "assistant-default",
        "assistant_message.default.yaml",
        "assistant_default",
    ),
    Template(
        "system",
        "system-builtin-and-custom-tools",
        "system_message.builtin_and_custom_tools.yaml",
        "system_message_builtin_and_custom_tools",
    ),
    Template(
        "system",
        "system-builtin-tools-only",
        "system_message.builtin_tools_only.yaml",
        "system_message_builtin_tools_only",
    ),
    Template(
        "system",
        "system-custom-tools-only",
        "system_message.custom_tools_only.yaml",
        "system_message_custom_tools_only",
    ),
    Template(
        "system",
        "system-default",
        "system_message.default.yaml",
        "system_default",
    ),
    Template(
        "tool",
        "tool-success",
        "tool_message.success.yaml",
        "tool_success",
        "Note ipython header and [stdout]",
    ),
    Template(
        "tool",
        "tool-failure",
        "tool_message.failure.yaml",
        "tool_failure",
        "Note ipython header and [stderr]",
    ),
]


class LLama31Interface:
    def __init__(self, tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json):
        self.tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(self.tokenizer)
        self.tool_prompt_format = tool_prompt_format

    def tool_response_messages(self, *args, **kwargs):
        template = ToolResponseGenerator().gen(*args, **kwargs)
        return ToolResponseMessage(
            call_id="call_id",
            tool_name="tool_name",
            content=template.render(),
        )

    def system_messages(
        self,
        builtin_tools: List[BuiltinTool],
        custom_tools: List[ToolDefinition],
        instruction: Optional[str] = None,
    ) -> List[Message]:
        messages = []

        default_gen = SystemDefaultGenerator()
        default_template = default_gen.gen()

        sys_content = ""

        tool_template = None
        if builtin_tools or custom_tools:
            tool_gen = BuiltinToolGenerator()
            tool_template = tool_gen.gen(builtin_tools + custom_tools)

            sys_content += tool_template.render()
            sys_content += "\n"

        sys_content += default_template.render()

        if instruction:
            sys_content += "\n\n"
            sys_content += instruction

        sys_content += "\n"
        messages.append(SystemMessage(content=sys_content))

        if custom_tools:
            if self.tool_prompt_format == ToolPromptFormat.json:
                tool_gen = JsonCustomToolGenerator()
            elif self.tool_prompt_format == ToolPromptFormat.function_tag:
                tool_gen = FunctionTagCustomToolGenerator()
            else:
                raise ValueError(
                    f"Non supported ToolPromptFormat {request.tool_prompt_format}"
                )

            custom_template = tool_gen.gen(custom_tools)
            messages.append(UserMessage(content=custom_template.render()))

        return messages

    def get_tokens(self, messages: List[Message]) -> List[int]:
        model_input = self.formatter.encode_dialog_prompt(
            messages,
            self.tool_prompt_format,
        )
        return model_input.tokens

    def assistant_response_messages(
        self,
        content: str,
        stop_reason: StopReason,
        tool_call: Optional[ToolCall] = None,
    ) -> List[CompletionMessage]:
        tool_calls = []
        if tool_call:
            tool_calls.append(tool_call)
        return [
            CompletionMessage(
                content=content,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
            )
        ]

    def user_message(self, content: str) -> List[UserMessage]:
        return [UserMessage(content=content)]

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

    def get_message_as_str_tokens(self, messages: Message) -> str:
        tokens = self.formatter.encode_dialog_prompt(messages)
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


def render_jinja_template_using_yaml(name: str):
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


def render_jinja_template(name: str, tool_prompt_format: ToolPromptFormat):
    by_name = {t.template_name: t for t in TEMPLATES}
    if name not in by_name:
        raise ValueError(f"No template found for `{name}`")

    template = by_name[name]
    interface = LLama31Interface(tool_prompt_format)

    if template.data_provider:
        data_func = getattr(template_data, template.data_provider)
        if template.role == "system":
            messages = interface.system_messages(**data_func())
        elif template.role == "tool":
            messages = interface.tool_response_messages(**data_func())
        elif template.role == "assistant":
            messages = interface.assistant_response_messages(**data_func())
        elif template.role == "user":
            messages = interface.user_message(**data_func())

        tokens = interface.get_tokens(messages)
        special_tokens = list(interface.tokenizer.special_tokens.values())
        tokens = [
            (interface.tokenizer.decode([t]), t in special_tokens) for t in tokens
        ]
        return template, tokens
    else:
        return render_jinja_template_using_yaml(name)
