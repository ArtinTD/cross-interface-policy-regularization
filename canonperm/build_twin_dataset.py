import argparse
import importlib.util
import json
import os
import random
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


MENU = _load("toolrl_menu")
TF = _load("twin_family")

LABEL_POOL = 100
THINK_DEFAULT = " I should use the appropriate tool with proper parameters to respond to the user's need. "


class RowUnusable(Exception):
    pass


def spans_of(text, tag):
    out, open_t, close_t, i = [], "<%s>" % tag, "</%s>" % tag, 0
    while True:
        s = text.find(open_t, i)
        if s < 0:
            return out
        e = text.find(close_t, s + len(open_t))
        if e < 0:
            return out
        out.append((s, e + len(close_t), text[s + len(open_t):e]))
        i = e + len(close_t)


def calls_in(body):
    body = body.strip()
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            return [json.loads(body)] if _loadable(body) else out
    return out


def _loadable(s):
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def gold_calls_of(output):
    out = []
    for _, _, body in spans_of(output, "tool_call"):
        for c in calls_in(body):
            if isinstance(c, dict) and c.get("name"):
                out.append(c)
    return out


def required_of(tools, gold_args_by_index, rng):
    sizes = [len({k for k in a if k in tools[i]["parameters"]})
             for i, a in gold_args_by_index.items()] or [1, 2]
    req = []
    for i, t in enumerate(tools):
        if i in gold_args_by_index:
            req.append({k for k in gold_args_by_index[i] if k in t["parameters"]})
            continue
        ks = list(t["parameters"])
        if not ks:
            req.append(set())
            continue
        rng.shuffle(ks)
        req.append(set(ks[:max(1, min(len(ks), rng.choice(sizes)))]))
    return req


def identity_map(n_menu):
    if n_menu >= LABEL_POOL:
        raise RowUnusable("menu of %d exceeds the %d-label pool" % (n_menu, LABEL_POOL))
    io = [-1] * LABEL_POOL
    for i in range(n_menu):
        io[i + 1] = i
    return io


def _canon_entries(tools):
    out, maps = [], []
    for t in tools:
        params, mp = TF.renumber_params(t["parameters"])
        out.append({"name": t["name"], "description": t["description"], "parameters": params})
        maps.append(mp)
    return out, maps


def _gold_text(golds, gold_indices, label_of, maps):
    if not golds:
        return None
    out = []
    for c, i in zip(golds, gold_indices):
        if i is None:
            out.append(json.dumps(c, ensure_ascii=False))
            continue
        mapped = {maps[i].get(k, k): v for k, v in (c.get("parameters") or {}).items()}
        out.append(json.dumps({"name": label_of[i], "parameters": mapped}, ensure_ascii=False))
    return "\n".join(out)


def render_clean(pre, tools, epi, segs, golds, gold_indices):
    entries, maps = _canon_entries(tools)
    labels = ["function_%02d" % (i + 1) for i in range(len(tools))]
    return {"sys": MENU.render_menu(pre, entries, epi, labels=labels),
            "user": render_history(segs, {i: (labels[i], maps[i]) for i in range(len(tools))}),
            "gold_call": _gold_text(golds, gold_indices, labels, maps),
            "identity_of": identity_map(len(tools)), "n_tools": len(tools)}


def draw_variant(pre, tools, epi, segs, golds, gold_indices, rng, spare_args=None):
    by_index = {}
    for c, i in zip(golds, gold_indices):
        if i is not None:
            by_index.setdefault(i, {}).update(c.get("parameters") or {})
    req = required_of(tools, by_index, rng)
    allow = set.intersection(*[TF.available_axes(t, req[i]) for i, t in enumerate(tools)])
    spares = _spares_per_tool(spare_args, tools, rng)

    entries, maps = _canon_entries(tools)
    slots, metas = [], []
    for i, t in enumerate(tools):
        d, m = TF.make_distractor(t, req[i], rng, donor=spares[i], allow=allow)
        slots.append((entries[i], d))
        metas.append(dict(m, source_index=i, is_reference=(i in by_index)))

    flat = [x for pair in slots for x in pair]
    order = list(range(len(flat)))
    rng.shuffle(order)
    new_pos = {o: n for n, o in enumerate(order)}
    labels = ["function_%02d" % (n + 1) for n in range(len(flat))]
    sys_text = MENU.render_menu(pre, [flat[o] for o in order], epi, labels=labels)

    io = [-1] * LABEL_POOL
    label_of = {}
    for i in range(len(tools)):
        pos = new_pos[2 * i]
        io[pos + 1] = i
        label_of[i] = labels[pos]

    return {"sys": sys_text,
            "user": render_history(segs, {i: (label_of[i], maps[i]) for i in range(len(tools))}),
            "gold": _gold_text(golds, gold_indices, label_of, maps),
            "identity_of": io, "mode": "twin", "n_tools": len(flat),
            "meta": {"axis": metas[next((i for i in by_index), 0)]["axis"] if by_index else "no_call"},
            "metas": metas}


class HistoryUnresolvable(RowUnusable):
    pass


def parse_history(user, tools):
    by_name = {}
    for i, t in enumerate(tools):
        by_name.setdefault(t["name"].strip(), i)

    segs, pos = [], 0
    spans = []
    for tag in ("tool_call", "obs"):
        spans += [(st, en, body, tag) for st, en, body in spans_of(user, tag)]
    spans.sort()
    for st, en, body, tag in spans:
        if st < pos:
            continue
        segs.append((None, user[pos:st]))
        lead = body[:len(body) - len(body.lstrip())]
        trail = body[len(body.rstrip()):]
        items = []
        for line in body.strip().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line.strip())
            except ValueError:
                items.append(("raw", line))
                continue
            group = obj if isinstance(obj, list) else [obj]
            out = []
            for o in group:
                if not isinstance(o, dict) or not o.get("name"):
                    out.append(("opaque", o))
                    continue
                i = by_name.get(str(o["name"]).strip())
                params = o.get("parameters")
                keys = list(params) if isinstance(params, dict) else []
                if i is None or any(k not in tools[i]["parameters"] for k in keys):
                    out.append(("opaque", o))
                else:
                    out.append(("call", i, o, keys))
            items.append(("json", obj, group, out, isinstance(obj, list)))
        segs.append((tag, lead, items, trail))
        pos = en
    segs.append((None, user[pos:]))
    return segs


def render_history(segs, chosen):
    out = []
    for seg in segs:
        if seg[0] is None:
            out.append(seg[1])
            continue
        tag, lead, items, trail = seg
        lines = []
        for it in items:
            if it[0] == "raw":
                lines.append(it[1])
                continue
            _, obj, group, resolved, is_list = it
            new_group = []
            for r in resolved:
                if r[0] == "opaque":
                    new_group.append(r[1])
                    continue
                _, i, o, keys = r
                label, amap = chosen[i]
                o2 = dict(o, name=label)
                if isinstance(o.get("parameters"), dict):
                    o2["parameters"] = {amap[k]: v for k, v in o["parameters"].items()}
                new_group.append(o2)
            lines.append(json.dumps(new_group if is_list else new_group[0], ensure_ascii=False))
        out.append("<%s>%s%s%s</%s>" % (tag, lead, "\n".join(lines), trail, tag))
    return "".join(out)


def collect_spare_args(data, rng, cap=4000):
    pool = {}
    for s in data:
        try:
            _, tools, _ = MENU.parse_menu(s["instruction"])
        except MENU.MenuFormatError:
            continue
        for t in tools:
            for k, v in t["parameters"].items():
                if isinstance(v, dict) and str(v.get("description", "")).strip():
                    pool.setdefault(json.dumps(v, sort_keys=True), (k, v))
    vals = list(pool.values())
    rng.shuffle(vals)
    return vals[:cap]


def _spares_per_tool(pool, tools, rng, per=3):
    if not pool:
        return [None] * len(tools)
    here = {json.dumps(v, sort_keys=True) for t in tools for v in t["parameters"].values()
            if isinstance(v, dict)}
    picked, seen, tries = [], set(), 0
    want = per * len(tools)
    while len(picked) < want and tries < want * 12:
        k, v = pool[rng.randrange(len(pool))]
        tries += 1
        j = json.dumps(v, sort_keys=True)
        if j not in here and j not in seen and k not in {kk for kk, _ in picked}:
            seen.add(j)
            picked.append((k, v))
    out = []
    for i in range(len(tools)):
        chunk = picked[i * per:(i + 1) * per]
        out.append({k: v for k, v in chunk} or None)
    return out


def make_fits(tokenizer, max_tokens, reserve):
    from tokenizers import Tokenizer
    if os.path.isdir(tokenizer):
        tk = Tokenizer.from_file(os.path.join(tokenizer, "tokenizer.json"))
    elif os.path.isfile(tokenizer):
        tk = Tokenizer.from_file(tokenizer)
    else:
        tk = Tokenizer.from_pretrained(tokenizer)
    budget = max_tokens - reserve

    def fits(v):
        return (len(tk.encode(v["sys"], add_special_tokens=False).ids)
                + len(tk.encode(v["user"], add_special_tokens=False).ids)) <= budget
    return fits


def build_row(sample, idx, split, rng, n_variants, fits=None, redraw_limit=12, spare_args=None):
    pre, tools, epi = MENU.parse_menu(sample["instruction"])
    user, output = sample["input"], sample["output"]
    names = [t["name"] for t in tools]

    segs = parse_history(user, tools)

    golds = gold_calls_of(output)
    gold_indices = [names.index(c["name"]) if c["name"] in names else None for c in golds]

    variants = []
    for _ in range(n_variants):
        for _ in range(redraw_limit):
            try:
                v = draw_variant(pre, tools, epi, segs, golds, gold_indices, rng,
                                 spare_args=spare_args)
            except TF.TwinError:
                raise
            if fits is None or fits(v):
                break
        else:
            raise RowUnusable("no variant fit the prompt budget in %d draws" % redraw_limit)
        variants.append(v)

    think = next((b for _, _, b in spans_of(output, "think")), THINK_DEFAULT)

    def wrap(call_text):
        return output if call_text is None else (
            "<think>%s</think>\n<tool_call>\n%s\n</tool_call>" % (think, call_text))

    clean = render_clean(pre, tools, epi, segs, golds, gold_indices)
    clean_gold = wrap(clean["gold_call"])
    for v in variants:
        v["gold"] = wrap(v["gold"])

    cp_meta = {"sys": clean["sys"], "user": clean["user"], "gold_call": clean_gold,
               "identity_of": clean["identity_of"], "n_tools": clean["n_tools"],
               "source": {"pre": pre, "epi": epi, "tools": tools, "golds": golds,
                          "gold_indices": gold_indices, "history": segs},
               "distractors": [{"sys": v["sys"], "user": v["user"], "gold": v["gold"],
                                "identity_of": v["identity_of"], "mode": v["mode"],
                                "n_tools": v["n_tools"],
                                "axis": v["meta"]["axis"]} for v in variants]}
    row = {"data_source": "rlla",
           "prompt": [{"role": "system", "content": clean["sys"]},
                      {"role": "user", "content": clean["user"]}],
           "ability": "math",
           "reward_model": {"style": "rule", "ground_truth": clean_gold},
           "extra_info": {"split": split, "index": idx, "interface": "twin_family_v5",
                          "n_tools": clean["n_tools"], "n_calls": len(golds),
                          "n_distractor": len(variants),
                          "cp_meta": json.dumps(cp_meta, ensure_ascii=False)}}
    return row, variants[0]["meta"]["axis"], variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlla-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="ToolRL's own held-out fraction (dataset/rlla_4k_raw/rlla.py: 2%)")
    ap.add_argument("--variants", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-prompt-tokens", type=int, default=4608,
                    help="what the run passes as PROMPT_LEN; twinning doubles the menu so ToolRL's own 2048 "
                         "left-truncates 6.9% of variants")
    ap.add_argument("--reserve", type=int, default=32, help="chat-template overhead held back")
    ap.add_argument("--redraw-limit", type=int, default=12)
    a = ap.parse_args()

    data = json.load(open(a.rlla_json))
    if a.limit:
        data = data[:a.limit]
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    fits = make_fits(a.tokenizer, a.max_prompt_tokens, a.reserve)
    spare = collect_spare_args(data, random.Random(a.seed + 17))
    print("replacement-argument pool: %d distinct arguments from across the corpus" % len(spare))
    rows, skip, axes = [], {}, {}
    for i, s in enumerate(data):
        try:
            row, axis, variants = build_row(s, i, "train", rng, a.variants,
                                            fits=fits, redraw_limit=a.redraw_limit, spare_args=spare)
        except (RowUnusable, MENU.MenuFormatError, TF.TwinError) as e:
            k = "%s: %s" % (type(e).__name__, re.sub(r"'.*?'", "'..'", str(e))[:60])
            skip[k] = skip.get(k, 0) + 1
            continue
        rows.append(row)
        axes[axis] = axes.get(axis, 0) + 1

    rng.shuffle(rows)
    ncut = max(1, int(len(rows) * a.val_frac)) if rows else 0
    test, train = rows[:ncut], rows[ncut:]
    for r in test:
        r["extra_info"]["split"] = "test"

    import pandas as pd
    for name, rs in (("train", train), ("test", test)):
        pd.DataFrame(rs).to_parquet(os.path.join(a.out_dir, name + ".parquet"))

    print("=== build_twin_dataset v3 (structures; two axes; %d variants/row) ===" % a.variants)
    print("rows in %d   skipped %s" % (len(data), skip or "{}"))
    print("axis mix (variant 0): %s" % axes)
    print("train %d   test %d" % (len(train), len(test)))
    print("out: %s" % a.out_dir)


if __name__ == "__main__":
    main()
