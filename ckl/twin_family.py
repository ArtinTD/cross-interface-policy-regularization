import json

INFERIORITY, SUFFICIENCY = "inferiority", "sufficiency"

P_SUFFICIENCY = 0.6


_INFERIOR = [
    "Deprecated; slower and kept only for backward compatibility.",
    "Legacy endpoint, noticeably slower than current ones.",
    "Deprecated. Scheduled for removal and slower to respond.",
    "Slower to respond, and may return partial results.",
    "Rate-limited to only a few requests per minute.",
    "Retained for older integrations; response times are poor.",
    "Unmaintained and slower; results may be stale.",
    "Slow path, intended only as a fallback for older clients.",
    "Deprecated: higher latency and no longer being updated.",
    "Older revision of this operation, slower and less reliable.",
]

class TwinError(Exception):
    pass


def _clone(tool):
    return {"name": tool["name"], "description": tool["description"],
            "parameters": json.loads(json.dumps(tool["parameters"]))}


def _identical(a, b):
    return (a["description"] == b["description"]
            and json.dumps(a["parameters"], sort_keys=True) == json.dumps(b["parameters"], sort_keys=True))


def renumber_params(params):
    out, mapping = {}, {}
    for i, (k, v) in enumerate(params.items(), 1):
        nk = "arg_%02d" % i
        out[nk] = v
        mapping[k] = nk
    return out, mapping


def canon_entries(tools):
    out = []
    for t in tools or []:
        params, _ = renumber_params(t.get("parameters") or {})
        out.append({"name": t.get("name", ""), "description": t.get("description", ""),
                    "parameters": params})
    return out

def available_axes(tool, required):
    axes = {INFERIORITY}
    if any(k in tool["parameters"] for k in required):
        axes.add(SUFFICIENCY)
    return axes


def _substitute_arg(params, victim, donor_params, rng):
    if not donor_params:
        return None
    vspec = json.dumps(params.get(victim), sort_keys=True)
    spare = [k for k in donor_params
             if k not in params and json.dumps(donor_params[k], sort_keys=True) != vspec]
    if not spare:
        return None
    take = rng.choice(spare)
    spec = donor_params[take]
    key, n = "_sub_%s" % take, 2
    while key in params:
        key, n = "_sub_%s_%d" % (take, n), n + 1
    out = {}
    for k, v in params.items():
        out[key if k == victim else k] = spec if k == victim else v
    return out


def make_distractor(tool, required, rng, donor=None, allow=None):
    allow = allow if allow is not None else available_axes(tool, required)
    d = _clone(tool)
    axis = SUFFICIENCY if (SUFFICIENCY in allow and rng.uniform(0, 1) < P_SUFFICIENCY) else INFERIORITY
    meta = {"axis": axis, "replaced": None}

    if axis == SUFFICIENCY:
        victims = [k for k in d["parameters"] if k in required
                   and isinstance(d["parameters"][k], dict)
                   and str(d["parameters"][k].get("description", "")).strip()]
        newp = _substitute_arg(d["parameters"], rng.choice(victims), donor, rng) if victims else None
        if newp is None:
            axis = INFERIORITY
        else:
            d["parameters"] = newp
            meta["replaced"] = True

    if axis == INFERIORITY:
        d["description"] = (tool["description"].rstrip() + " " + rng.choice(_INFERIOR)).strip()

    meta["axis"] = axis
    d["parameters"], meta["arg_map"] = renumber_params(d["parameters"])
    if len(d["parameters"]) != len(tool["parameters"]):
        raise TwinError("distractor changed the argument count (%d vs %d)"
                        % (len(d["parameters"]), len(tool["parameters"])))
    return d, meta
