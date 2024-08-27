# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import textwrap

from .base import PromptTemplate, PromptTemplateGeneratorBase


class AssistantResponseGenerator(PromptTemplateGeneratorBase):

    def gen(self, message: CompletionMessage):
        # This is the output format for custom tools in json
        # """<|python_tag|>{
        #     "type": "function",
        #     "name": "get_boiling_point",
        #     "parameters": {
        #         "liquid_name": "polyjuice",
        #         "celcius": "true"
        #     }
        # }"""
        # and this is with `function_tag`
        # <function="{{ tool_call.tool_name }}">{{ tool_call.arguments | tojson }}</function>

        template_str = textwrap.dedent(
            """
            {%- set builtins = ["brave_search", "wolfram_alpha", "photogen"] -%}
            {%- if tool_call -%}
            <|python_tag|>
            {%- if tool_call.tool_name in builtins -%}
            {{ tool_call.tool_name }}.call(query="{{ tool_call.arguments['query'] }}")
            {%- else -%}
            {%- if tool_prompt_format == json -%}
            {
                "type": "function",
                "name": "{{ tool_call.tool_name }}",
                "paramerters: {{ tool_call.arguments | tojson(indent=4) }}
            {%- else -%}
            <function="{{ tool_call.tool_name }}">{{ tool_call.arguments | tojson }}</function>
            {%- endif -%}
            {%- else -%}
            {{ content }}
            {%- endif -%}
            {%- if end_of_message %}<|eom_id|>{% else %}<|eot_id|>{% endif %}
            """
        )
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            data = {"tool_call": tool_call.model_dump()}
        else:
            data = {"content": message.content}

        data["end_of_message"] = message.end_of_message == StopReason.end_of_message

        return PromptTemplate(template_str.lstrip("\n"), data)

    def data_examples(self):
        return []
