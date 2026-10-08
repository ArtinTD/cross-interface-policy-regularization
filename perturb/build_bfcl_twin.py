import argparse
import copy
import json
import os
import random
import sys

CODE = os.environ.get("CODE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CODE, "canonperm"))
sys.path.insert(0, os.path.join(CODE, "perturb"))

import build_twin_dataset as BT
import canonicalize_bfcl as CB
import rbtc16 as R
import twin_family as TF


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def properties_of(fn):
    p = (fn.get("parameters") or {}).get("properties")
    return p if isinstance(p, dict) else {}


def needed_args(allowed_by_arg):
    return {k: v for k, v in (allowed_by_arg or {}).items() if isinstance(v, list) and "" not in v}


def harness_safe(spec):
    if not (isinstance(spec, dict) and str(spec.get("description", "")).strip()):
        return False
    if "array" in str(spec.get("type", "")).lower() or "list" in str(spec.get("type", "")).lower():
        items = spec.get("items")
        return isinstance(items, dict) and "type" in items
    return True


def collect_spare_args(records, rng, cap=4000):
    pool = {}
    for r in records:
        for fn in r.get("function") or []:
            for k, v in properties_of(fn).items():
                if harness_safe(v):
                    pool.setdefault(json.dumps(v, sort_keys=True), (k, v))
    vals = list(pool.values())
    rng.shuffle(vals)
    return vals[:cap]


def rename_self(node, pattern, labels):
    if isinstance(node, list):
        return [rename_self(x, pattern, labels) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "description" and isinstance(v, str):
            out[k] = CB.relabel_names(v, pattern, labels)[0]
        else:
            out[k] = rename_self(v, pattern, labels)
    return out


def twin_function(fn, tool, d, meta, twin_name):
    inv = {new: old for old, new in meta["arg_map"].items()}
    props = {inv[k]: v for k, v in d["parameters"].items()}
    orig_keys, twin_keys = list(tool["parameters"]), list(props)
    if len(orig_keys) != len(twin_keys):
        raise TF.TwinError("argument count changed")
    at_same_position = dict(zip(orig_keys, twin_keys))
    twin = copy.deepcopy(fn)
    params = twin.get("parameters") if isinstance(twin.get("parameters"), dict) else {}
    params["properties"] = props
    for key in ("required", "optional"):
        if isinstance(params.get(key), list):
            params[key] = [at_same_position.get(x, x) for x in params[key]]
    twin["parameters"] = params
    twin["name"] = twin_name
    twin["description"] = d["description"]
    pattern = CB.name_pattern([fn["name"]])
    return rename_self(twin, pattern, {fn["name"]: twin_name})


def twin_record(record, gold, pool, seed):
    fns = record.get("function") or []
    names = [f.get("name") for f in fns]
    tools = [{"name": f["name"], "description": f.get("description") or "", "parameters": properties_of(f)}
             for f in fns]
    by_index = {}
    for call in gold or []:
        if isinstance(call, dict) and len(call) == 1:
            (fn_name, args), = call.items()
            if fn_name in names:
                by_index.setdefault(names.index(fn_name), {}).update(needed_args(args))
    rng = random.Random("%d:%s" % (seed, record["id"]))
    req = BT.required_of(tools, by_index, rng)
    allow = set.intersection(*[TF.available_axes(t, req[i]) for i, t in enumerate(tools)])
    spares = BT._spares_per_tool(pool, tools, rng)

    used = set(names)
    flat, metas = [], []
    for i, (fn, tool) in enumerate(zip(fns, tools)):
        d, m = TF.make_distractor(tool, req[i], rng, donor=spares[i], allow=allow)
        twin_name, n = fn["name"] + "_alt", 2
        while twin_name in used:
            twin_name, n = "%s_alt%d" % (fn["name"], n), n + 1
        used.add(twin_name)
        flat += [copy.deepcopy(fn), twin_function(fn, tool, d, m, twin_name)]
        metas.append({"axis": m["axis"], "is_reference": i in by_index})

    order = list(range(len(flat)))
    rng.shuffle(order)
    new_pos = {o: n for n, o in enumerate(order)}
    out = dict(record)
    out["function"] = [flat[o] for o in order]
    out[R.INJECTED_KEY] = sorted(new_pos[2 * i + 1] for i in range(len(fns)))
    pairs = [dict(metas[i], source_index=i,
                  real_label="function_%02d" % (new_pos[2 * i] + 1),
                  twin_label="function_%02d" % (new_pos[2 * i + 1] + 1))
             for i in range(len(fns))]
    return out, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(x for x in os.listdir(a.clean) if x.startswith("BFCL") and x.endswith(".json"))
    data = {f: load_jsonl(os.path.join(a.clean, f)) for f in files}
    pools = {}
    for lang in sorted({R.language_of(f) for f in files}):
        recs = [r for f in files if R.language_of(f) == lang for r in data[f]]
        pools[lang] = collect_spare_args(recs, random.Random(a.seed + 17))
        print("replacement-argument pool %-10s %d" % (lang, len(pools[lang])))

    src = os.path.join(a.out, "pert_normal", "twin")
    dst = os.path.join(a.out, "pert_canon", "twin")
    axes = {}
    for f in files:
        gold = load_jsonl(os.path.join(a.clean, "possible_answer", f))
        gold_by_id = {g["id"]: g.get("ground_truth", []) for g in gold}
        rows, meta = [], []
        for r in data[f]:
            nr, pairs = twin_record(r, gold_by_id.get(r["id"], []), pools[R.language_of(f)], a.seed)
            rows.append(nr)
            meta.append({"id": r["id"], "pairs": pairs})
            for p in pairs:
                if p["is_reference"]:
                    axes[p["axis"]] = axes.get(p["axis"], 0) + 1
        write_jsonl(os.path.join(src, f), rows)
        write_jsonl(os.path.join(src, "possible_answer", f), gold)
        write_jsonl(os.path.join(a.out, "twin_meta", f.replace(".json", ".jsonl")), meta)
        status, n, ng, stats = CB.convert_file(src, dst, f)
        print("%-40s %s %4d records %4d gold  name replacements: question %d, description %d"
              % (f, status, n, ng, stats.get("question", 0), stats.get("description", 0)))
    print("axis of the gold tools' twins:", axes)


if __name__ == "__main__":
    main()
