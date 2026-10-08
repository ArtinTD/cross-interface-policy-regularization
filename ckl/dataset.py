import json
import random
import re

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from . import twin_candidates, twin_family
from .twin_family import canon_entries
from .interface_perm import (fc_convert_gold, fc_convert_surface, fc_draw_view,
                             fc_residual_system, fc_tools_from_menu)
from .label_index import menu_numbers
from .rowbuild import MAX_MENU

LABEL_POOL = 100
def _pad(values, width=MAX_MENU, fill=-1):
    if len(values) > width:
        raise ValueError("%d entries exceed the %d-wide index column; raise MAX_MENU" % (len(values), width))
    return torch.tensor(list(values) + [fill] * (width - len(values)), dtype=torch.long)


def is_fc(text):
    return "<function=" in (text or "")


def to_fc_gold(text, what):
    if is_fc(text):
        raise ValueError("%s is already on the FC surface, so converting it again would parse `<function=` "
                         "as JSON. Convert once, at the boundary where the surface is known." % what)
    return fc_convert_gold(text)


_THINK = re.compile(r"<think>.*?</think>", re.S)


def think_prefix(gold):
    m = _THINK.search(gold or "")
    return m.group(0) if m else ""


def clean_fc(cp_meta):
    src = cp_meta.get("source") or {}
    tools, n = src.get("tools"), int(cp_meta.get("n_tools") or 0)
    if not tools or n <= 0:
        return None
    numbers = menu_numbers(cp_meta.get("identity_of") or [], n)
    if numbers is None:
        return None
    gold, kind = to_fc_gold(cp_meta.get("gold_call") or "", "the clean gold")
    if gold is None:
        return None
    return {"sys": fc_residual_system(src["pre"], src["epi"]),
            "user": fc_convert_surface(cp_meta.get("user") or ""),
            "tools": fc_tools_from_menu(canon_entries(tools), ["function_%02d" % m for m in numbers]),
            "gold": gold, "numbers": numbers, "n_tools": n, "kind": kind}


def added_fc(cp_meta, rng, clean_gold):
    src = cp_meta.get("source")
    if not src:
        return None
    cand = twin_candidates.draw_candidates(src, 1, rng, twin_family=twin_family,
                                           label_pool=LABEL_POOL, fc=True)
    if not cand:
        return None
    view = fc_draw_view(cand[0], cand[0]["identity_of"], rng)
    n = int(cp_meta.get("n_tools") or 0)
    numbers = menu_numbers(view.get("identity_of") or [], n)
    if not n or numbers is None:
        return None
    span = (view.get("gold") or "").strip()
    if span and not is_fc(span):
        raise ValueError("the twin-menu gold is not on the FC surface: draw_candidates was called without "
                         "fc=True, so the menu and the gold disagree about the surface")
    if not span:
        gold = clean_gold
    else:
        think = think_prefix(clean_gold)
        gold = ("%s\n%s" % (think, span)) if think else span
    return {"sys": view.get("sys") or cand[0]["sys"], "user": view.get("user") or cand[0]["user"],
            "tools": view.get("tools") or cand[0]["tools"], "gold": gold,
            "numbers": numbers, "n_tools": n}


class CKLTwinDataset(RLHFDataset):

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        super().__init__(data_files, tokenizer, config, processor=processor, max_samples=max_samples)
        ckl_cfg = config.get("ckl", {}) or {}
        self.twin_interface = bool(ckl_cfg.get("twin_interface", True))
        self.max_add_draws = int(ckl_cfg.get("max_add_draws", 4))
        self.seed = int(ckl_cfg.get("seed", 0))
        self.n_no_add = 0
        self.n_no_clean = 0
        self.n_clean_error = 0
        self.n_add_error = 0
        self._said = set()

    def _guarded(self, what, fn, *a):
        try:
            return fn(*a)
        except Exception as e:
            if what == "clean interface":
                self.n_clean_error += 1
            else:
                self.n_add_error += 1
            if what not in self._said:
                self._said.add(what)
                print("ckl.dataset: a row's %s could not be built, and rows like it will be counted rather "
                      "than raised (%s: %s)" % (what, type(e).__name__, e))
            return None

    def _fits(self, messages, tools):
        text = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                  add_generation_prompt=True)
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) <= self.max_prompt_length

    def __getitem__(self, item):
        row = super().__getitem__(item)
        extra = row.get("extra_info") or {}
        raw = extra.get("cp_meta")
        cp = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(cp, dict):
            raise ValueError("row %s carries no cp_meta: this dataset consumes a twin parquet built by "
                             "canonperm/build_twin_dataset.py" % item)
        row["extra_info"] = {k: v for k, v in extra.items() if k != "cp_meta"}

        clean = self._guarded("clean interface", clean_fc, cp)
        if clean is None:
            self.n_no_clean += 1
            row.update(ckl_readable=torch.tensor(0, dtype=torch.long),
                       ckl_numbers=_pad([]), ckl_n_tools=torch.tensor(0, dtype=torch.long))
        else:
            row["raw_prompt"] = [{"role": "system", "content": clean["sys"]},
                                 {"role": "user", "content": clean["user"]}]
            row["ckl_tools"] = clean["tools"]
            rm = row.get("reward_model")
            row["reward_model"] = dict(rm or {}, style="rule", ground_truth=clean["gold"])
            row.update(ckl_readable=torch.tensor(1, dtype=torch.long),
                       ckl_numbers=_pad(clean["numbers"]),
                       ckl_n_tools=torch.tensor(clean["n_tools"], dtype=torch.long))

        row["ckl_interface"] = torch.tensor(0, dtype=torch.long)
        row["ckl_pair_id"] = torch.tensor(int(extra.get("index", item)), dtype=torch.long)

        add = None
        if self.twin_interface and clean is not None:
            rng = random.Random((self.seed * 1000003) ^ (int(item) * 2654435761)
                                ^ int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
            for _ in range(max(1, self.max_add_draws)):
                cand = self._guarded("twin interface", added_fc, cp, rng, clean["gold"])
                if cand is None:
                    continue
                msgs = [{"role": "system", "content": cand["sys"]},
                        {"role": "user", "content": cand["user"]}]
                if self._fits(msgs, cand["tools"]):
                    add = dict(cand, messages=msgs)
                    break
        if add is None:
            self.n_no_add += 1
            row.update(ckl_twin_ok=torch.tensor(0, dtype=torch.long),
                       ckl_twin_prompt=[], ckl_twin_tools=[], ckl_twin_gold="",
                       ckl_twin_numbers=_pad([]), ckl_twin_n_tools=torch.tensor(0, dtype=torch.long))
        else:
            row.update(ckl_twin_ok=torch.tensor(1, dtype=torch.long),
                       ckl_twin_prompt=add["messages"], ckl_twin_tools=add["tools"],
                       ckl_twin_gold=add["gold"], ckl_twin_numbers=_pad(add["numbers"]),
                       ckl_twin_n_tools=torch.tensor(add["n_tools"], dtype=torch.long))
        return row
