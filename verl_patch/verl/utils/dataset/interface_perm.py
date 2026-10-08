import json
import os
import re

_NAME_LINE = re.compile(r"(?m)^(\d+\.\s*Name:\s*)(.+?)\s*$")
_FUNC = re.compile(r"\bfunction_(\d{2})\b")
_TOOLCALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_N_LABELS = int(os.environ.get('IDENTITY_LABEL_SPACE', '100'))
_N_ARG_LABELS = _N_LABELS


def renumber_functions(text, mapping):
    return _FUNC.sub(lambda m: "function_%02d" % mapping[int(m.group(1))], text)


def tool_args(sys_text):
    dec = json.JSONDecoder()
    out, ms = [], list(_NAME_LINE.finditer(sys_text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(sys_text)
        pm = re.search(r"Parameters:\s*", sys_text[m.end():end])
        keys = []
        if pm:
            s = m.end() + pm.end()
            if s < len(sys_text) and sys_text[s] == "{":
                try:
                    obj, _ = dec.raw_decode(sys_text, s)
                    if isinstance(obj, dict):
                        keys = list(obj.keys())
                except ValueError:
                    pass
        out.append((m.group(2).strip(), keys))
    return out


def relabel_params(sys_text, arg_by_func):
    dec = json.JSONDecoder()
    ms = list(_NAME_LINE.finditer(sys_text))
    pieces, i = [], 0
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(sys_text)
        pm = re.search(r"Parameters:\s*", sys_text[m.end():end])
        if pm is None:
            continue
        s = m.end() + pm.end()
        if s >= len(sys_text) or sys_text[s] != "{":
            continue
        try:
            obj, stop = dec.raw_decode(sys_text, s)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        amap = arg_by_func.get(m.group(2).strip(), {})
        pieces.append(sys_text[i:s])
        pieces.append(json.dumps({amap.get(a, a): v for a, v in obj.items()}, ensure_ascii=False))
        i = stop
    pieces.append(sys_text[i:])
    return "".join(pieces)


def relabel_call_params(text, arg_by_func):
    dec = json.JSONDecoder()

    def fix(mb):
        body, out, i = mb.group(1), [], 0
        while i < len(body):
            j = body.find("{", i)
            if j < 0:
                out.append(body[i:])
                break
            out.append(body[i:j])
            try:
                obj, stop = dec.raw_decode(body, j)
            except ValueError:
                out.append(body[j])
                i = j + 1
                continue
            if isinstance(obj, dict) and isinstance(obj.get("parameters"), dict):
                amap = arg_by_func.get(obj.get("name"), {})
                obj["parameters"] = {amap.get(a, a): v for a, v in obj["parameters"].items()}
                out.append(json.dumps(obj, ensure_ascii=False))
            else:
                out.append(body[j:stop])
            i = stop
        return "<tool_call>" + "".join(out) + "</tool_call>"

    return _TOOLCALL.sub(fix, text)


def draw_view(template, identity_of, rng, permute_args=True):
    n_data = max(_N_LABELS, len(identity_of or []))
    if _N_LABELS >= n_data:
        perm = list(range(_N_LABELS))
        rng.shuffle(perm)
    else:
        perm = list(range(n_data))
        used = sorted({int(m.group(1)) for m in _FUNC.finditer(template["sys"])})
        if len(used) > _N_LABELS:
            raise ValueError("the menu declares %d tools but the alphabet has %d labels"
                             % (len(used), _N_LABELS))
        for u, t in zip(used, rng.sample(range(_N_LABELS), len(used))):
            perm[u] = t
    sys_t = renumber_functions(template["sys"], perm)
    user_t = renumber_functions(template.get("user", ""), perm)
    gold_t = renumber_functions(template.get("gold", ""), perm)
    ident = [-1] * n_data
    for n, p in enumerate(identity_of or []):
        if p is not None and int(p) >= 0:
            ident[perm[n]] = int(p)
    if permute_args:
        amap = {}
        for label, keys in tool_args(sys_t):
            keys = list(dict.fromkeys(keys))
            if not keys:
                continue
            if len(keys) >= _N_ARG_LABELS:
                raise ValueError("tool %s declares %d arguments but only %d argument labels can be rendered"
                                 % (label, len(keys), _N_ARG_LABELS))
            order = list(range(len(keys)))
            rng.shuffle(order)
            amap[label] = {k: "arg_%02d" % (order[i] + 1) for i, k in enumerate(keys)}
        if amap:
            sys_t = relabel_params(sys_t, amap)
            gold_t = relabel_call_params(gold_t, amap)
    return {"sys": sys_t, "user": user_t, "gold": gold_t, "identity_of": ident}

def _last_params_end(sys_text):
    dec = json.JSONDecoder()
    end = None
    for m in re.finditer(r"Parameters:\s*", sys_text):
        s = m.end()
        if s < len(sys_text) and sys_text[s] == "{":
            try:
                _, e = dec.raw_decode(sys_text, s)
                end = e
            except ValueError:
                pass
    return end


def _split_tool_region(sys_text):
    ms = list(_NAME_LINE.finditer(sys_text))
    lpe = _last_params_end(sys_text)
    if not ms or lpe is None:
        return None
    blocks = [sys_text[m.start():(ms[i + 1].start() if i + 1 < len(ms) else lpe)].rstrip("\n")
              for i, m in enumerate(ms)]
    return sys_text[:ms[0].start()], blocks, sys_text[lpe:]


def _renumber_block(block, new_num):
    return re.sub(r"^\d+\.", "%d." % new_num, block, count=1)


def _permute_params(block, rng):
    dec = json.JSONDecoder()
    m = re.search(r"Parameters:\s*", block)
    if not m:
        return block
    s = m.end()
    if s >= len(block) or block[s] != "{":
        return block
    try:
        obj, e = dec.raw_decode(block, s)
    except ValueError:
        return block
    if not isinstance(obj, dict) or len(obj) < 2:
        return block
    keys = list(obj.keys())
    rng.shuffle(keys)
    return block[:s] + json.dumps({k: obj[k] for k in keys}, ensure_ascii=False) + block[e:]


def reorder_view(sys_text, rng):
    parts = _split_tool_region(sys_text)
    if parts is None:
        return None
    head, blocks, tail = parts
    n = len(blocks)
    for _ in range(32):
        perm = list(range(n))
        rng.shuffle(perm)
        out = head + "\n".join(_renumber_block(_permute_params(blocks[src], rng), pos + 1)
                               for pos, src in enumerate(perm)) + tail
        if out != sys_text:
            return out
    return None






