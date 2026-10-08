import json
import os
import re

_FUNC = re.compile(r"\bfunction_(\d{2})\b")
_N_LABELS = int(os.environ.get('IDENTITY_LABEL_SPACE', '100'))
_N_ARG_LABELS = _N_LABELS


def renumber_functions(text, mapping):
    return _FUNC.sub(lambda m: "function_%02d" % mapping[int(m.group(1))], text)


_FC_FUNC_TAG = re.compile(r"(<function=)([^>]*?)(\s*>)")


def _fc_name(tool):
    return (tool.get("function") or {}).get("name", "")


def _fc_props(tool):
    p = (tool.get("function") or {}).get("parameters") or {}
    q = p.get("properties")
    return q if isinstance(q, dict) else {}


def _fc_set_props(tool, props):
    tool["function"]["parameters"]["properties"] = props


def _fc_copy(tool):
    f = tool.get("function") or {}
    p = f.get("parameters") or {}
    return dict(tool, function=dict(f, parameters=dict(p, properties=dict(p.get("properties") or {}))))


def fc_map_gold(gold, name_map=None, arg_maps=None):
    def one(m):
        head, old, tail = m.group(1), m.group(2), m.group(3)
        return head + (name_map.get(old, old) if name_map else old) + tail

    if not gold:
        return gold
    out, pos = [], 0
    for m in _FC_FUNC_TAG.finditer(gold):
        out.append(gold[pos:m.start()])
        old = m.group(2)
        out.append(one(m))
        end = gold.find("</function>", m.end())
        end = len(gold) if end < 0 else end
        body = gold[m.end():end]
        amap = (arg_maps or {}).get(old) or {}
        if amap:
            body = re.sub(r"(<parameter=)([^>\s]+)(\s*>)",
                          lambda p: p.group(1) + amap.get(p.group(2), p.group(2)) + p.group(3), body)
        out.append(body)
        pos = end
    out.append(gold[pos:])
    return "".join(out)


def fc_draw_view(template, identity_of, rng, permute_args=True):
    tools = [_fc_copy(t) for t in template["tools"]]
    used = sorted({int(m.group(1)) for t in tools
                   for m in [_FUNC.search(_fc_name(t))] if m})
    n_data = max(_N_LABELS, len(identity_of or []))
    if _N_LABELS >= n_data:
        perm = list(range(_N_LABELS))
        rng.shuffle(perm)
    else:
        perm = list(range(n_data))
        if len(used) > _N_LABELS:
            raise ValueError("the menu declares %d tools but the alphabet has %d labels"
                             % (len(used), _N_LABELS))
        for u, t in zip(used, rng.sample(range(_N_LABELS), len(used))):
            perm[u] = t

    name_map = {}
    for t in tools:
        old = _fc_name(t)
        m = _FUNC.search(old)
        if m:
            new = _FUNC.sub(lambda _m: "function_%02d" % perm[int(_m.group(1))], old)
            name_map[old] = new
            t["function"]["name"] = new

    arg_maps = {}
    if permute_args:
        for t in tools:
            old_name = next((k for k, v in name_map.items() if v == _fc_name(t)), _fc_name(t))
            props = _fc_props(t)
            keys = list(props)
            if not keys:
                continue
            if len(keys) >= _N_ARG_LABELS:
                raise ValueError("tool %s declares %d arguments but only %d argument labels can be rendered"
                                 % (_fc_name(t), len(keys), _N_ARG_LABELS))
            order = list(range(len(keys)))
            rng.shuffle(order)
            amap = {k: "arg_%02d" % (order[i] + 1) for i, k in enumerate(keys)}
            arg_maps[old_name] = amap
            _fc_set_props(t, {amap[k]: v for k, v in props.items()})

    ident = [-1] * n_data
    for n, p in enumerate(identity_of or []):
        if p is not None and int(p) >= 0:
            ident[perm[n]] = int(p)
    return {"sys": renumber_functions(template.get("sys", ""), perm),
            "user": renumber_functions(template.get("user", ""), perm),
            "gold": fc_map_gold(template.get("gold", ""), name_map, arg_maps),
            "tools": tools, "identity_of": ident}


_AVAIL = "**Available Tools**"
_DROP_SECTIONS = ("**Output Format**", "**Important Notes**")
_JSON_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_RESPONSE = re.compile(r"<response>(.*?)</response>", re.DOTALL)


def fc_tool_spec(name, description, properties):
    return {"type": "function", "function": {
        "name": name,
        "description": description or "",
        "parameters": {"type": "object",
                       "properties": properties if isinstance(properties, dict) else {},
                       "required": []}}}


def fc_tools_from_menu(entries, labels=None):
    return [fc_tool_spec(labels[i] if labels is not None else e.get("name", ""),
                         e.get("description", ""), e.get("parameters"))
            for i, e in enumerate(entries)]


def fc_residual_system(pre, epi):
    i = pre.find(_AVAIL)
    if i < 0:
        raise ValueError("system prose carries no %s heading, so the menu cannot be lifted out of it"
                         % _AVAIL)
    head = pre[:i].rstrip()
    tail = epi
    for marker in _DROP_SECTIONS:
        j = tail.find(marker)
        if j < 0:
            continue
        nxt = min((q for q in (tail.find(m, j + len(marker)) for m in _DROP_SECTIONS) if q > 0),
                  default=-1)
        tail = tail[:j] + (tail[nxt:] if nxt > 0 else "")
    return (head + "\n\n" + tail.strip()).strip()


def _fc_value_text(v):
    if isinstance(v, dict) or (isinstance(v, (list, tuple)) and not isinstance(v, str)):
        return json.dumps(v)
    return "" if v is None else str(v)


def fc_render_call(name, params):
    out = ["<tool_call>\n<function=%s>\n" % name]
    for k, v in (params or {}).items():
        out.append("<parameter=%s>\n%s\n</parameter>\n" % (k, _fc_value_text(v)))
    out.append("</function>\n</tool_call>")
    return "".join(out)


def fc_calls_from_json(text):
    out = []
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        for x in (o if isinstance(o, list) else [o]):
            if not isinstance(x, dict) or "name" not in x:
                raise ValueError("call object without a name field: %r" % line[:80])
            out.append((x["name"], x.get("parameters") or x.get("arguments") or {}))
    return out


def fc_render_calls(text):
    return "\n".join(fc_render_call(n, p) for n, p in fc_calls_from_json(text))


def fc_convert_surface(text):
    t = text or ""
    out, pos = [], 0
    for m in _JSON_CALL.finditer(t):
        out.append(t[pos:m.start()])
        out.append(fc_render_calls(m.group(1)))
        pos = m.end()
    out.append(t[pos:])
    return _RESPONSE.sub(lambda m: m.group(1).strip(), "".join(out))


def fc_convert_gold(gold_text):
    t = gold_text or ""
    has_call = _JSON_CALL.search(t) is not None
    if has_call and _RESPONSE.search(t):
        return None, "both"
    if not has_call and t.strip()[:1] in ("{", "["):
        return fc_render_calls(t), "call"
    return fc_convert_surface(t).strip(), ("call" if has_call else "plain")
