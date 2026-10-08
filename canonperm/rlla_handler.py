import json
import re

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override

_TC = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.S)
_XML_FN = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|\Z)", re.S)
_XML_ARG = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)(?:</parameter>|\Z)", re.S)


def _coerce(text):
    t = text.strip()
    try:
        return json.loads(t)
    except (ValueError, TypeError):
        return t


def call_objects(result):
    out = []
    for block in _TC.findall(result or ""):
        block = block.strip()
        if not block:
            continue
        got = []
        try:
            obj = json.loads(block)
            got = obj if isinstance(obj, list) else [obj]
        except (ValueError, TypeError):
            got = []
        if not got:
            for line in block.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                got += obj if isinstance(obj, list) else [obj]
        if not got:
            for name, body in _XML_FN.findall(block):
                got.append({"name": name.strip(),
                            "parameters": {k.strip(): _coerce(v) for k, v in _XML_ARG.findall(body)}})
        for obj in got:
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            if name is None or str(name).strip().lower() == "none":
                continue
            params = obj.get("parameters")
            if params is None:
                params = obj.get("arguments") or {}
            out.append({"name": str(name), "parameters": params if isinstance(params, dict) else {}})
    return out

json_string = """{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}
{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}"""

SYS = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{json_string}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).
"""


class RLLAHandler(OSSHandler):
    @override
    def _format_prompt(self, messages, function, turn_type="single_turn"):
        def convert_to_format_tool(tools, count=1):
            if isinstance(tools, dict):
                ft = {
                    "name": tools["name"],
                    "description": tools["description"],
                    "parameters": tools["parameters"].get("properties", {}),
                }
                return (f"{count}. Name: {ft['name']}\nDescription: {ft['description']}\n"
                        f"Parameters: {json.dumps(ft['parameters'])}")
            elif isinstance(tools, list):
                return "\n".join(convert_to_format_tool(t, i + 1) for i, t in enumerate(tools))
            return tools

        tools = convert_to_format_tool(function)
        SYSTEM_PROMPT = SYS.format(tools=tools, json_string=json_string)
        USER_PROMPT = "**Dialogue Records History**\n"
        for msg_idx, message in enumerate(messages):
            if message["role"] == "system":
                continue
            elif message["role"] == "user":
                if turn_type == "multi_turn":
                    USER_PROMPT += (f"<user> {message['content'].strip()}\nUse the one or more necessary tool calls to "
                        "complete the task. You could perform tool calls for multiple rounds so you can try and error. "
                        "Please make a comprehensive plan about how to achieve the goal step by step, and begin to call "
                        "the tool step by step. If no tools apply or required parameters are missing, please also directly "
                        "state this in your response without tool calls. </user>\n")
                else:
                    USER_PROMPT += (f"<user> {message['content'].strip()}\nIf there's no appropriate tools to apply or "
                        "required parameters are missing, please directly inform me in your response without any tool call, "
                        "or call the tool with the name as 'None'. Otherwise, you should use one or more necessary tool "
                        "calls to complete the given task in this turn. </user>\n")
            elif message["role"] == "tool":
                tool_name = message["name"].strip()
                tool_result = message["content"].strip()
                USER_PROMPT += f"<obs> You have made the tool call {tool_name}. Execution returns: {tool_result} </obs>\n"
                if msg_idx == len(messages) - 1:
                    USER_PROMPT += ("\n<user> If you think you have completed the current task, or the task cannot be "
                        "finished, please respond directly without additional tool calls. If you encounter an error during "
                        "tool execution or the task remains unfinished, retry with the one or more necessary tool calls "
                        "according to your thought and plan until completion. Based on the tool execution feedback, reflect "
                        "on if understanding or selectioin of tool is wrong, what tool calling step is missing, and how to "
                        "achieve the task goal from now on. </user>\n")
            elif message["role"] == "assistant":
                message["content"] = message["content"].strip()
                USER_PROMPT += f"\n{message['content'].strip()}\n"
        USER_PROMPT = USER_PROMPT.strip()
        return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n")

    @override
    def decode_ast(self, result, language="Python", has_tool_call_tag=False):
        if "<tool_call>" not in result:
            return []
        return [{c["name"]: c["parameters"]} for c in call_objects(result)]

    @override
    def decode_execute(self, result, has_tool_call_tag=False):
        if "<tool_call>" not in result:
            return []
        return self.xlam_json_to_python_tool_calls(call_objects(result))

    @staticmethod
    def xlam_json_to_python_tool_calls(tool_calls):
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        python_format = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                name = tool_call.get("name", "")
                arguments = tool_call.get("parameters", {})
                if name.strip().lower() == "none":
                    continue
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
                python_format.append(f"{name}({args_str})")
        return python_format
