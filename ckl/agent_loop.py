from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


@register("ckl_single_turn")
class CKLSingleTurnAgentLoop(SingleTurnAgentLoop):

    async def run(self, sampling_params, **kwargs):
        self._ckl_tools = kwargs.get("ckl_tools") or None
        return await super().run(sampling_params, **kwargs)

    async def apply_chat_template(self, messages, tools=None, **kwargs):
        return await super().apply_chat_template(messages, tools=tools or getattr(self, "_ckl_tools", None),
                                                 **kwargs)
