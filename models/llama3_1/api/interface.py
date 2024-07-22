import json
from datetime import datetime

from typing import List, Optional

import fire
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


def sample_custom_tools() -> List[ToolDefinition]:
    return [
        ToolDefinition(
            tool_name="get_boiling_point",
            description="Returns the boiling point of a liquid",
            parameters={
                "liquid_name": ToolParamDefinition(
                    param_type="string",
                    description="The name of the liquid",
                    required=True,
                ),
                "celcius": ToolParamDefinition(
                    param_type="boolean",
                    description="Whether to return the boiling point in Celcius",
                    required=False,
                ),
            },
        ),
        ToolDefinition(
            tool_name="trending_songs",
            description="Returns the trending songs on a Music site",
            parameters={
                "country": ToolParamDefinition(
                    param_type="string",
                    description="The country to return trending songs for",
                    required=True,
                ),
                "n": ToolParamDefinition(
                    param_type="int",
                    description="The number of songs to return",
                    required=False,
                ),
            },
        ),
    ]


def jin():
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("api/templates"))
    template = env.get_template("system_message.jinja")

    current_date = datetime.now().strftime("%d %B %Y")
    context = {
        'builtin_tools': ["brave_search", "code_interpreter"],
        "custom_tools": [t.dict() for t in sample_custom_tools()],
        "today": current_date,
    }
    cprint("System message", "yellow")
    cprint("-" * 100, "yellow")
    print(template.render(context))

    cprint("User message", "yellow")
    cprint("-" * 100, "yellow")
    context = {
        "content": "Tell me a story"
    }
    print(env.get_template("user_message.jinja").render(context))

    cprint("Tool response message", "yellow")
    cprint("-" * 100, "yellow")
    context = {
        "is_builtin_tool": True,
        "status": "success",
        "stdout": "The top 100 most popular songs in the US are: Foo, bar and baz",
        # "stderr": "Oops",
    }
    print(env.get_template("tool_message.jinja").render(context))

    cprint("Tool call", "yellow")
    cprint("-" * 100, "yellow")
    context = {
        "content": "Hello how are you doing today? How can I help you?",
        "end_of_message": True,
        "tool_call": {
            "tool_name": "trending_songs",
            "arguments": {
                "query": "What are the top 100 most popular songs in the US?"
            },
        }
    }
    print(env.get_template("assistant_message.jinja").render(context))


def main(tokenizer_path: str):
    llama = LLama3_1_Interface(tokenizer_path)

    system_message = llama.recommended_system_message(
        builtin_tools=[],
        custom_tools=[],
    )
    cprint("Default system prompt", "green")
    print(llama.get_message_as_str_tokens(system_message))
    print("\n")

    cprint("System prompt with all builtin tools", "green")
    system_message = llama.recommended_system_message(
        builtin_tools=[
            BuiltinTool.brave_search,
            BuiltinTool.wolfram_alpha,
            BuiltinTool.photogen,
            BuiltinTool.code_interpreter,
        ],
        custom_tools=[],
    )
    print(llama.get_message_as_str_tokens(system_message))
    print("\n")

    cprint("System prompt with only custom tools", "green")
    system_message = llama.recommended_system_message(
        builtin_tools=[],
        custom_tools=sample_custom_tools(),
    )
    print(llama.get_message_as_str_tokens(system_message))
    print("\n")

    cprint("Sample builtin tool call from model", "green")
    print(llama.get_message_as_str_tokens(llama.get_sample_builtin_tool_call_message()))
    print("\n")


if __name__ == "__main__":
    fire.Fire(jin)
