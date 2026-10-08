import copy
import hashlib
import json
import os
import re

DISPLAY = {
    "query_typos": "Typo",
    "redundant": "RedunTool",
    "same_name_A": "Dup-NoDesc",
    "same_name_B": "Dup-Desc",
    "same_name_C": "Dup-WrongP",
    "same_name_D": "Dup-DescWP",
    "same_name_E": "Dup-SwapDP",
    "cost_decoy": "Cost decoy",
    "cost_decoy_nt": "Cost decoy (nat. lang.)",
    "cost_decoy_abbrev": "Cost decoy (abbrev)",
    "latency_decoy": "Latency decoy",
    "latency_decoy_nt": "Latency decoy (nat. lang.)",
    "latency_decoy_abbrev": "Latency decoy (abbrev)",
}
TYPES = list(DISPLAY)
_SAME_NAME = {"same_name_A", "same_name_B", "same_name_C", "same_name_D", "same_name_E"}
_COST = {"cost_decoy", "cost_decoy_nt", "cost_decoy_abbrev"}
_LAT = {"latency_decoy", "latency_decoy_nt", "latency_decoy_abbrev"}
_REWARD = _COST | _LAT
RENAME_TYPES = set(_REWARD)

_ADJ = {"a": "s", "s": "d", "d": "f", "e": "r", "r": "t", "t": "y", "o": "i", "i": "o",
        "n": "m", "m": "n", "c": "v", "l": "k", "u": "y", "g": "h", "p": "o"}
_COST_INSTR = ["Please use a cost-effective option.", "Choose the option with minimal cost.",
               "I have a limited budget for this task."]
_LAT_INSTR = ["Use the tool with the shortest response time.", "Please use a fast option."]

_COST_INSTR_CLEAR = ("Each tool's description states how many dollars one call to it costs. Complete the task "
                     "while spending as few dollars as possible: whenever two tools would both do the job, "
                     "call the cheaper one.")
_LAT_INSTR_CLEAR = ("Each tool's description states how many milliseconds one call to it takes. Complete the "
                    "task in as little total time as possible: whenever two tools would both do the job, "
                    "call the faster one.")


_INVENTORY = ("Before making a decision, give a short record of every tool, what it does, and its important "
              "properties.")

INSTRUCTION_STYLES = ("default", "budget", "rbtc", "inventory", "inventory+budget")


def instruction_for(ptype, style, rng):
    if style not in INSTRUCTION_STYLES:
        raise ValueError("unknown decoy instruction style %r; one of %s"
                         % (style, ", ".join(INSTRUCTION_STYLES)))
    is_cost = ptype in _COST
    short = list(_COST_INSTR if is_cost else _LAT_INSTR)
    objective = _COST_INSTR_CLEAR if is_cost else _LAT_INSTR_CLEAR
    if style == "default":
        return rng.choice(short) if ptype.endswith("_abbrev") else objective
    if style == "rbtc":
        return rng.choice(short)
    if style == "budget":
        return objective
    if style == "inventory":
        return _INVENTORY
    return _INVENTORY + " " + objective


INSTRUCTIONS = list(_COST_INSTR) + list(_LAT_INSTR) + [_COST_INSTR_CLEAR, _LAT_INSTR_CLEAR, _INVENTORY]

OBJECTIVE_INSTRUCTIONS = list(_COST_INSTR) + list(_LAT_INSTR) + [_COST_INSTR_CLEAR, _LAT_INSTR_CLEAR]


def states_preference(text):
    t = text or ""
    return any(i in t for i in INSTRUCTIONS)


def states_objective(text):
    t = text or ""
    return any(i in t for i in OBJECTIVE_INSTRUCTIONS)


def _rng(task_id, ptype):
    import random
    return random.Random(int(hashlib.md5(("%s|%s" % (task_id, ptype)).encode()).hexdigest()[:12], 16))


def _typo_word(w, rng):
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    mode = rng.choice(["swap", "adj", "drop", "dup"])
    if mode == "swap":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if mode == "adj" and w[i].lower() in _ADJ:
        return w[:i] + _ADJ[w[i].lower()] + w[i + 1:]
    if mode == "drop":
        return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


def _typo_text(text, rng, rate=0.2):
    out = []
    for tok in re.split(r"(\s+)", text):
        if tok.strip() and tok.isalpha() and rng.random() < rate:
            out.append(_typo_word(tok, rng))
        else:
            out.append(tok)
    return "".join(out)


def _abbrev(name):
    parts = [p for p in re.split(r"[._]", name) if p]
    ab = "_".join(p[:3] for p in parts) if parts else name[:4]
    return ab if ab != name else name + "_x"


def _redun_name(gold_name, donor_name):
    sep = "." if "." in gold_name else "_"
    gseg = [p for p in re.split(r"[._]", gold_name) if p]
    dseg = [p for p in re.split(r"[._]", donor_name) if p]
    if len(gseg) >= 2 and dseg:
        return sep.join(gseg[:-1] + [dseg[-1]])
    if dseg and gseg:
        return sep.join([gseg[0], dseg[-1]])
    return gold_name + sep + "alt"


def _fn_name(t):
    return t.get("name")


def _empty_params():
    return {"type": "dict", "properties": {}, "required": []}


def _gold_fn_names(gold):
    names = []
    for call in gold or []:
        if isinstance(call, dict):
            names += list(call.keys())
    return names


def _find(functions, name):
    for t in functions:
        if _fn_name(t) == name:
            return t
    return None


def _targets(functions, gold):
    names = [n for n in _gold_fn_names(gold) if _find(functions, n)]
    if not names and functions:
        names = [_fn_name(functions[0])]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


INJECTED_KEY = "_injected"


def _twin(base, name=None, description=None, parameters=None, injected=True):
    t = copy.deepcopy(base)
    if name is not None:
        t["name"] = name
    t["description"] = "" if description is None else description
    if parameters is not None:
        t["parameters"] = copy.deepcopy(parameters)
    if injected:
        t[INJECTED_KEY] = True
    else:
        t.pop(INJECTED_KEY, None)
    return t


def _insert_before(functions, pairs):
    out = list(functions)
    for nm, tool in pairs:
        pos = next((j for j, t in enumerate(out) if _fn_name(t) == nm), None)
        if pos is None:
            out.append(tool)
        else:
            out.insert(pos, tool)
    return out


def _psig(t):
    return json.dumps((t or {}).get("parameters"), sort_keys=True)


def _fully_labelable(t):
    def ok(v):
        if not isinstance(v, dict):
            return True
        has_props = isinstance(v.get("properties"), dict)
        for key in ("required", "optional"):
            val = v.get(key)
            if isinstance(val, list) and val and not has_props:
                return False
        if has_props and not all(ok(s) for s in v["properties"].values()):
            return False
        it = v.get("items")
        if isinstance(it, dict):
            return ok(it)
        if isinstance(it, list):
            return all(ok(x) for x in it)
        return True

    params = (t or {}).get("parameters")
    if not isinstance(params, dict):
        return True
    return all(ok(s) for s in (params.get("properties") or {}).values())


def language_of(name):
    if "javascript" in name:
        return "javascript"
    if "java" in name:
        return "java"
    return "python"


_DONOR_SOURCES = {"python": ("multiple", "parallel", "parallel_multiple"),
                  "java": ("simple_java",),
                  "javascript": ("simple_javascript",)}


def build_donor_pool(src_data_dir, language="python"):
    seen, pool = set(), []
    for cat in _DONOR_SOURCES[language]:
        path = os.path.join(src_data_dir, "BFCL_v4_%s.json" % cat)
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            for t in json.loads(line).get("function", []):
                key = (t.get("name"), t.get("description"), _psig(t))
                if key not in seen and _fully_labelable(t):
                    seen.add(key)
                    pool.append(t)
    pool.sort(key=lambda t: (t.get("name") or "", _psig(t)))
    return pool


def _pick_donor(rng, avoid, in_pool, corpus, need_desc=False):
    bad_p = {_psig(t) for t in avoid}
    bad_d = {t.get("description") or "" for t in avoid}
    for pool in (in_pool, corpus or []):
        cand = [t for t in pool if _psig(t) not in bad_p
                and (not need_desc or (t.get("description") or "") not in bad_d)]
        if cand:
            return cand[rng.randrange(len(cand))]
    return None


def _relabel_gold(gold, namemap):
    out = []
    for call in gold:
        if isinstance(call, dict) and len(call) == 1:
            (fn, args), = call.items()
            out.append({namemap.get(fn, fn): args})
        else:
            out.append(call)
    return out


def perturb(ptype, functions, question, gold, task_id, donors=None, decoy_first=False,
            dup_first=False, instruction="default"):
    functions = copy.deepcopy(functions or [])
    question = copy.deepcopy(question or [])
    gold = copy.deepcopy(gold or [])
    rng = _rng(task_id, ptype)
    tgt = _targets(functions, gold)

    if ptype == "query_typos":
        for turn in question:
            for m in turn:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    m["content"] = _typo_text(m["content"], rng)
        return functions, question, gold

    if ptype == "redundant":
        goldtools = [t for t in (_find(functions, n) for n in tgt) if t is not None]
        others = [t for t in functions if _fn_name(t) not in tgt]
        add, used, pairs = [], set(), []
        for n in tgt:
            base = _find(functions, n)
            if base is None:
                continue
            src = _pick_donor(rng, goldtools, [o for o in others if _fn_name(o) not in used], donors,
                              need_desc=True)
            if src is None:
                continue
            used.add(_fn_name(src))
            nm = _redun_name(n, _fn_name(src))
            while _find(functions, nm) or _find(add, nm):
                nm += "_x"
            twin = _twin(src, name=nm, description=src.get("description", ""),
                         parameters=src.get("parameters"))
            add.append(twin)
            pairs.append((n, twin))
        if dup_first:
            return _insert_before(functions, pairs), question, gold
        return functions + add, question, gold

    if ptype in _SAME_NAME:
        add, pairs = [], []
        for n in tgt:
            base = _find(functions, n)
            if base is None:
                continue
            donor = None
            if ptype in ("same_name_C", "same_name_D", "same_name_E"):
                donor = _pick_donor(rng, [base], [t for t in functions if _fn_name(t) not in tgt],
                                    donors, need_desc=(ptype == "same_name_E"))
                if donor is None:
                    continue
            if ptype == "same_name_A":
                desc, params = "", _empty_params()
            elif ptype == "same_name_B":
                desc, params = base.get("description", ""), _empty_params()
            elif ptype == "same_name_C":
                desc, params = "", donor.get("parameters")
            elif ptype == "same_name_D":
                desc, params = base.get("description", ""), donor.get("parameters")
            else:
                desc, params = donor.get("description", ""), donor.get("parameters")
            twin = _twin(base, name=n, description=desc, parameters=params)
            add.append(twin)
            pairs.append((n, twin))
        if dup_first:
            return _insert_before(functions, pairs), question, gold
        return functions + add, question, gold

    if ptype in _REWARD:
        is_cost = ptype in _COST
        is_nt = ptype.endswith("_nt")
        is_ab = ptype.endswith("_abbrev")

        def tag(val):
            if is_cost:
                return ("(costs approximately $%.3f per call)" % val) if is_nt else ("[Cost: $%.3f/call]" % val)
            return ("(typically responds in %dms)" % val) if is_nt else \
                   (("[Latency: %dms]" % val) if is_ab else ("[Response time: ~%dms]" % val))

        def lo():
            return round(rng.uniform(0.002, 0.02), 3) if is_cost else rng.randint(20, 80)

        def hi():
            return round(rng.uniform(0.08, 0.25), 3) if is_cost else rng.randint(800, 3000)

        def apptag(desc, val):
            return ("%s %s" % (tag(val), desc)) if is_ab else ("%s %s" % (desc, tag(val)))

        newfns, namemap = [], {}
        for t in functions:
            n = _fn_name(t)
            d = t.get("description") or ""
            if n in tgt and n not in namemap:
                gn = _abbrev(n) if is_ab else n + "_1"
                while _find(functions, gn) or gn in namemap.values():
                    gn += "_x"
                cheap = _twin(t, name=gn, description=apptag(d, lo()), parameters=t.get("parameters"),
                              injected=False)
                dear = _twin(t, name=n, description=apptag(d, hi()), parameters=t.get("parameters"),
                             injected=True)
                newfns += [dear, cheap] if decoy_first else [cheap, dear]
                namemap[n] = gn
            else:
                newfns.append(_twin(t, name=n, description=apptag(d, hi()), parameters=t.get("parameters"),
                                    injected=False))
        instr = instruction_for(ptype, instruction or "default", rng)
        for turn in reversed(question):
            hit = False
            for m in reversed(turn):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    m["content"] = m["content"].rstrip() + " " + instr
                    hit = True
                    break
            if hit:
                break
        return newfns, question, _relabel_gold(gold, namemap)

    raise ValueError("unknown perturbation type: %s" % ptype)
