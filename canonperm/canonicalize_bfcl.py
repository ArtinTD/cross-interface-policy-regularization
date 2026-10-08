import argparse
import json
import os
import re

QUOTES = "'\"`"


def load_records(path):
    if not os.path.exists(path):
        return []
    txt = open(path).read()
    recs, jsonl = [], True
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            jsonl = False
            break
    if jsonl and recs:
        return recs
    obj = json.loads(txt)
    return obj if isinstance(obj, list) else [obj]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_category(stem, listing):
    pat = re.compile(r"^(?:BFCL_v\d+_)?" + re.escape(stem) + r"\.json$")
    return [x for x in listing if pat.match(x)]


def is_single_turn_gold(gold):
    if not isinstance(gold, list):
        return False
    for g in gold:
        if isinstance(g, dict):
            return True
        if isinstance(g, (list, str)):
            return False
    return True


def name_labels(functions):
    out = {}
    for f in functions:
        nm = f.get("name") if isinstance(f, dict) else None
        if isinstance(nm, str) and nm not in out:
            out[nm] = "function_%02d" % (len(out) + 1)
    return out


def arg_labels(properties):
    if not isinstance(properties, dict):
        return {}
    return {k: "arg_%02d" % (i + 1) for i, k in enumerate(properties)}


def nested_properties(schema):
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if isinstance(props, dict):
        return props
    items = schema.get("items")
    if isinstance(items, dict) and isinstance(items.get("properties"), dict):
        return items["properties"]
    return None


_CAMEL = re.compile(r"[a-z][A-Z]")


def is_identifier_name(name):
    return ("." in name or "_" in name or any(c.isdigit() for c in name)
            or _CAMEL.search(name) is not None)


def name_pattern(names):
    ss = sorted({n for n in names if n}, key=len, reverse=True)
    if not ss:
        return None
    return re.compile(r"(?<![A-Za-z0-9_.])(?:" + "|".join(re.escape(s) for s in ss)
                      + r")(?![A-Za-z0-9_]|\.[A-Za-z0-9_])")


def relabel_names(text, pattern, labels):
    if not isinstance(text, str) or pattern is None:
        return text, 0, []
    out, pos, n, skipped = [], 0, 0, []
    for m in pattern.finditer(text):
        name = m.group(0)
        lead = text[m.start() - 1] if m.start() > 0 else ""
        trail = text[m.end()] if m.end() < len(text) else ""
        quoted = bool(lead) and lead in QUOTES and trail == lead
        if not (is_identifier_name(name) or quoted):
            skipped.append(name)
            continue
        out.append(text[pos:m.start()])
        out.append(labels[name])
        pos = m.end()
        n += 1
    out.append(text[pos:])
    return "".join(out), n, skipped


def canon_schema(node, relabel):
    if isinstance(node, list):
        total, out = 0, []
        for x in node:
            cx, n = canon_schema(x, relabel)
            out.append(cx)
            total += n
        return out, total
    if not isinstance(node, dict):
        return node, 0
    out, total = dict(node), 0
    if isinstance(out.get("description"), str):
        out["description"], n = relabel(out["description"])
        total += n
    props = out.get("properties")
    if isinstance(props, dict):
        amap = arg_labels(props)
        newprops = {}
        for k, v in props.items():
            newprops[amap[k]], n = canon_schema(v, relabel)
            total += n
        out["properties"] = newprops
        for key in ("required", "optional"):
            if isinstance(out.get(key), list):
                out[key] = [amap.get(x, x) for x in out[key]]
    if isinstance(out.get("items"), (dict, list)):
        out["items"], n = canon_schema(out["items"], relabel)
        total += n
    return out, total


def canon_functions(functions, labels, relabel):
    out, total = [], 0
    for f in functions:
        if not isinstance(f, dict):
            out.append(f)
            continue
        nf = dict(f)
        nm = nf.get("name")
        if isinstance(nm, str):
            nf["name"] = labels.get(nm, nm)
        if isinstance(nf.get("description"), str):
            nf["description"], n = relabel(nf["description"])
            total += n
        if isinstance(nf.get("parameters"), dict):
            nf["parameters"], n = canon_schema(nf["parameters"], relabel)
            total += n
        out.append(nf)
    return out, total


def canon_gold_value(value, schema):
    props = nested_properties(schema)
    if props is None:
        return value
    if isinstance(value, list):
        return [canon_gold_value(x, schema) for x in value]
    if isinstance(value, dict):
        amap = arg_labels(props)
        return {amap.get(k, k): canon_gold_value(v, props.get(k)) for k, v in value.items()}
    return value


def _gold_instance(instances, fn, injected):
    cands = instances.get(fn) or []
    if not cands:
        return None
    real = [t for i, t in cands if i not in injected]
    return real[0] if real else cands[0][1]


def canon_gold(gold, labels, instances, injected=()):
    out = []
    for call in gold or []:
        if not (isinstance(call, dict) and len(call) == 1):
            out.append(call)
            continue
        (fn, args), = call.items()
        props = ((_gold_instance(instances, fn, injected) or {}).get("parameters") or {}).get("properties") or {}
        amap = arg_labels(props)
        if isinstance(args, dict):
            args = {amap.get(k, k): canon_gold_value(v, props.get(k)) for k, v in args.items()}
        out.append({labels.get(fn, fn): args})
    return out


def canon_function_list(functions):
    labels = name_labels(functions)
    pattern = name_pattern(labels)

    def relabel(text):
        new, n, _skipped = relabel_names(text, pattern, labels)
        return new, n

    out, _n = canon_functions(functions, labels, relabel)
    inv_name = {lab: nm for nm, lab in labels.items()}
    inv_arg = {}
    for f in functions:
        nm = f.get("name") if isinstance(f, dict) else None
        lab = labels.get(nm)
        if lab is None or lab in inv_arg:
            continue
        props = (f.get("parameters") or {}).get("properties") or {}
        inv_arg[lab] = {v: k for k, v in arg_labels(props).items()}
    return out, inv_name, inv_arg


def canon_record(record):
    functions = record.get("function") or []
    labels = name_labels(functions)
    instances = {}
    for i_fn, f in enumerate(functions):
        nm = f.get("name") if isinstance(f, dict) else None
        if isinstance(nm, str):
            instances.setdefault(nm, []).append((i_fn, f))
    pattern = name_pattern(labels)

    def relabel(text):
        new, n, _skipped = relabel_names(text, pattern, labels)
        return new, n

    new = dict(record)
    new["function"], n_desc = canon_functions(functions, labels, relabel)

    n_q, skipped = 0, []
    question = record.get("question")
    if isinstance(question, list):
        turns = []
        for turn in question:
            if not isinstance(turn, list):
                turns.append(turn)
                continue
            msgs = []
            for m in turn:
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    content, n, sk = relabel_names(m["content"], pattern, labels)
                    m = {**m, "content": content}
                    n_q += n
                    skipped.extend(sk)
                msgs.append(m)
            turns.append(msgs)
        new["question"] = turns
    return new, labels, instances, {"question": n_q, "description": n_desc, "skipped": skipped}


def convert_file(src_data_dir, out_dir, fname):
    records = load_records(os.path.join(src_data_dir, fname))
    gold_by_id = {g["id"]: g.get("ground_truth", [])
                  for g in load_records(os.path.join(src_data_dir, "possible_answer", fname))
                  if isinstance(g, dict) and "id" in g}
    sample = next((gold_by_id[r["id"]] for r in records if r.get("id") in gold_by_id), [])
    if gold_by_id and not is_single_turn_gold(sample):
        return "skip_multi_turn", len(records), len(gold_by_id), {}

    out_records, out_gold = [], []
    stats = {"question": 0, "description": 0, "residual": {}}
    for r in records:
        new, labels, instances, s = canon_record(r)
        out_records.append(new)
        stats["question"] += s["question"]
        stats["description"] += s["description"]
        if s["skipped"]:
            stats["residual"][r.get("id")] = sorted(set(s["skipped"]))
        if r.get("id") in gold_by_id:
            out_gold.append({"id": r["id"],
                             "ground_truth": canon_gold(gold_by_id[r["id"]], labels, instances,
                                                       set(r.get("_injected") or ()))})

    write_jsonl(os.path.join(out_dir, fname), out_records)
    if out_gold:
        write_jsonl(os.path.join(out_dir, "possible_answer", fname), out_gold)
    return "ok", len(out_records), len(out_gold), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--categories", default="", help="comma-separated BFCL category file stems; "
                    "default = all BFCL_*.json in SRC_DATA_DIR")
    args = ap.parse_args()

    listing = sorted(x for x in os.listdir(args.src_data_dir) if x.endswith(".json"))
    if args.categories:
        files = []
        for stem in (s.strip() for s in args.categories.split(",")):
            if not stem:
                continue
            hits = resolve_category(stem, listing)
            if len(hits) == 1:
                files.append(hits[0])
            else:
                print(f"{'UNRESOLVED':16s} {stem}  (exact matches: {hits or 'none'})")
    else:
        files = [x for x in listing if x.startswith("BFCL")]

    totals = {"question": 0, "description": 0}
    residual = {}
    for fname in files:
        try:
            status, n, ng, stats = convert_file(args.src_data_dir, args.out_dir, fname)
        except Exception as e:
            print(f"{'ERROR':16s} {fname}  ({type(e).__name__}: {str(e)[:100]})")
            continue
        print(f"{status:16s} {fname}  ({n} records, {ng} gold)")
        for k in totals:
            totals[k] += stats.get(k, 0)
        residual.update(stats.get("residual", {}))

    print(f"tool-name replacements: {totals['question']} in questions, "
          f"{totals['description']} in descriptions")
    print(f"not replaced (declared tool name used as a plain English word in the question): "
          f"{len(residual)}")
    for rid, names in residual.items():
        print(f"  {rid}  {', '.join(names)}")


if __name__ == "__main__":
    main()
