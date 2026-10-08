import json

from .interface_perm import (fc_convert_surface, fc_render_calls, fc_residual_system,
                                               fc_tools_from_menu)


def _render(pre, entries, epi, labels):
    out = [pre]
    for i, e in enumerate(entries):
        out.append("%d. Name: %s\nDescription: %s\nParameters: %s\n"
                   % (i + 1, labels[i], e["description"], json.dumps(e["parameters"])))
    out.append(epi)
    return "".join(out)


def draw_candidates(source, k, rng, spare_args=None, twin_family=None, label_pool=100,
                    fc=False):
    if twin_family is None:
        return []
    TF = twin_family
    tools = source["tools"]
    golds, gidx = source["golds"], source["gold_indices"]

    by_index = {}
    for c, i in zip(golds, gidx):
        if i is not None:
            by_index.setdefault(i, {}).update(c.get("parameters") or {})

    out = []
    for _ in range(k):
        sizes = [len({q for q in a if q in tools[i]["parameters"]}) for i, a in by_index.items()] or [1, 2]
        req = []
        for i, t in enumerate(tools):
            if i in by_index:
                req.append({q for q in by_index[i] if q in t["parameters"]})
                continue
            ks = list(t["parameters"])
            rng.shuffle(ks)
            req.append(set(ks[:max(1, min(len(ks), rng.choice(sizes)))]) if ks else set())

        allow = set.intersection(*[TF.available_axes(t, req[i]) for i, t in enumerate(tools)])

        entries, maps, slots, axes = [], [], [], []
        for i, t in enumerate(tools):
            params, mp = TF.renumber_params(t["parameters"])
            entries.append({"description": t["description"], "parameters": params})
            maps.append(mp)
        for i, t in enumerate(tools):
            spare = None
            try:
                d, m = TF.make_distractor(t, req[i], rng, donor=spare, allow=allow)
            except TF.TwinError:
                continue
            slots.append((i, d))
            axes.append(m["axis"])
        if not slots:
            continue

        flat, owner = [], []
        for i, e in enumerate(entries):
            flat.append(e)
            owner.append(i)
        for i, d in slots:
            flat.append({"description": d["description"], "parameters": d["parameters"]})
            owner.append(-1 - i)
        order = list(range(len(flat)))
        rng.shuffle(order)
        labels = ["function_%02d" % (n + 1) for n in range(len(flat))]

        io = [-1] * label_pool
        label_of = {}
        for new_pos, old in enumerate(order):
            if owner[old] >= 0:
                io[new_pos + 1] = owner[old]
                label_of[owner[old]] = labels[new_pos]

        gold_txt = None
        if golds:
            parts = []
            for c, i in zip(golds, gidx):
                if i is None or i not in label_of:
                    parts.append(json.dumps(c, ensure_ascii=False))
                    continue
                mapped = {maps[i].get(q, q): v for q, v in (c.get("parameters") or {}).items()}
                parts.append(json.dumps({"name": label_of[i], "parameters": mapped}, ensure_ascii=False))
            gold_txt = "\n".join(parts)
            if fc:
                gold_txt = fc_render_calls(gold_txt)

        entries_in_order = [flat[o] for o in order]
        cand = {"user": render_history(source["history"], label_of, maps, fc=fc),
                "gold": gold_txt, "identity_of": io, "n_tools": len(flat),
                "mode": "twin_adv", "axes": axes}
        if fc:
            cand["sys"] = fc_residual_system(source["pre"], source["epi"])
            cand["tools"] = fc_tools_from_menu(entries_in_order, labels)
        else:
            cand["sys"] = _render(source["pre"], entries_in_order, source["epi"], labels)
        out.append(cand)
    return out


def render_history(segs, label_of, maps, fc=False):
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
                if i not in label_of:
                    new_group.append(o)
                    continue
                o2 = dict(o, name=label_of[i])
                if isinstance(o.get("parameters"), dict):
                    o2["parameters"] = {maps[i].get(q, q): v for q, v in o["parameters"].items()}
                new_group.append(o2)
            lines.append(json.dumps(new_group if is_list else new_group[0], ensure_ascii=False))
        out.append("<%s>%s%s%s</%s>" % (tag, lead, "\n".join(lines), trail, tag))
    text = "".join(out)
    return fc_convert_surface(text) if fc else text


