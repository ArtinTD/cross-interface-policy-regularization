import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_RLLA = os.path.normpath(os.path.join(_HERE, "..", "verl_patch", "verl", "utils", "reward_score", "rlla.py"))
_MOD = None


def scorer():
    global _MOD
    if _MOD is None:
        if not os.path.isfile(_RLLA):
            raise FileNotFoundError(
                "the ToolRL scorer is not at %s. It is loaded from the vendored-verl overlay on purpose, so "
                "both arms score with one copy -- see this module's docstring." % _RLLA)
        spec = importlib.util.spec_from_file_location("ckl_rlla_reward", _RLLA)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MOD = m
    return _MOD


THINK_OPEN = "<think>\n"


def as_scored_before(response):
    text = response or ""
    return text if text.lstrip().startswith("<think>") else THINK_OPEN + text


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    step = int((extra_info or {}).get("global_steps", 0) or 0)
    score, fmt, correctness, length = scorer().compute_score(as_scored_before(solution_str), ground_truth,
                                                             step=step)
    return {"score": float(score), "format_score": float(fmt),
            "correctness_score": float(correctness), "length_score": float(length)}
