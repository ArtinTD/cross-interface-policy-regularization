
VENDOR = "ckl"

_FUSED_HEAD = {}

_ARMED = {"on": False}

_TABLES = {}


def _tokens_for(engine):
    from .tokens import build

    tk = engine.model_config.tokenizer
    key = id(tk)
    if key not in _TABLES:
        _TABLES[key] = build(tk)
    return _TABLES[key]


def _gate_id(engine):
    return _tokens_for(engine)[1].gate_id


def _table(engine):
    return _tokens_for(engine)[0]


def register():
    if getattr(register, "_done", False):
        return
    from verl.workers.engine.base import EngineRegistry
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    from . import engine as ckl_engine

    ckl_engine.make_engine(FSDPEngineWithLMHead, EngineRegistry, _gate_id, _table, take_fused_head,
                           arm_fused_head, vendor=VENDOR)

    patch_fused_head()
    register._done = True


def patch_fused_head():
    import torch

    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if getattr(patch_fused_head, "_done", False):
        return
    original = FusedLinearForPPO.forward

    def wrapper(self, hidden_states, vocab_weights, input_ids, temperature=1.0):
        if torch.is_grad_enabled() or _ARMED["on"]:
            _FUSED_HEAD["hidden"], _FUSED_HEAD["weight"] = hidden_states, vocab_weights
        return original(self, hidden_states, vocab_weights, input_ids, temperature)

    FusedLinearForPPO.forward = wrapper
    patch_fused_head._done = True


def arm_fused_head(on):
    _ARMED["on"] = bool(on)


def take_fused_head():
    return _FUSED_HEAD.pop("hidden", None), _FUSED_HEAD.pop("weight", None)


register()


