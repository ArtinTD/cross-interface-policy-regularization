import json
import time

from overrides import override

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool
from bfcl_eval.utils import contain_multi_turn_interaction


class QwenFCHandler(OSSHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS

    @override
    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        if contain_multi_turn_interaction(test_entry["id"]):
            raise NotImplementedError(
                "QwenFCHandler is single-turn only; %s is a multi-turn category. Implement "
                "_add_assistant_message_FC / _add_execution_results_FC before running it." % test_entry["id"])
        return self.inference_single_turn_FC(test_entry, include_input_log)


    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["tools"] = convert_to_tool(
            test_entry["function"], GORILLA_TO_OPENAPI, self.model_style)
        return inference_data

    @override
    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        kwargs = {
            "model": self.model_path_or_id,
            "messages": inference_data["message"],
            "temperature": self.temperature,
            "max_tokens": 4096,
            "timeout": 72000,
        }
        if inference_data["tools"]:
            kwargs["tools"] = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": inference_data["message"],
                                                 "tools": inference_data["tools"]}
        start = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        return api_response, time.time() - start

    @override
    def _parse_query_response_FC(self, api_response) -> dict:
        message = api_response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])
        if calls:
            model_responses = [{c.function.name: c.function.arguments} for c in calls]
        else:
            model_responses = message.content or ""
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": message,
            "tool_call_ids": [c.id for c in calls],
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }


    @override
    def decode_ast(self, result, language="Python", has_tool_call_tag=False):
        if not isinstance(result, list):
            return []
        out = []
        for item in result:
            if not isinstance(item, dict):
                continue
            for name, args in item.items():
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except json.JSONDecodeError:
                    parsed = {}
                out.append({name: parsed})
        return out

    @override
    def decode_execute(self, result, has_tool_call_tag=False):
        if not isinstance(result, list):
            return []
        return convert_to_function_call(result)
