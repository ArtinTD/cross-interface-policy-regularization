#!/usr/bin/env python3
import argparse
import glob
import json
import os


def rows(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


INJECTED_KEY = "_injected"


def prompt_of(rec):
    return {"messages": [m for turn in (rec.get("question") or [])
                         for m in (turn if isinstance(turn, list) else [turn])],
            "tools": rec.get("function") or [],
            "injected": rec.get(INJECTED_KEY) or []}


def query_of(rec):
    q = rec.get("question") or []
    msgs = [m for turn in q for m in (turn if isinstance(turn, list) else [turn])]
    user = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if user:
        return "\n".join(user)
    return json.dumps(msgs) if msgs else ""


def review_cell(cell, model, data_dir, parts=()):
    written = {}
    rdir = os.path.join(cell, "result", model)
    for rpath in sorted(glob.glob(os.path.join(rdir, "**", "BFCL_v4_*_result.json"), recursive=True)):
        part = os.path.basename(rpath)[len("BFCL_v4_"):-len("_result.json")]
        if parts and part not in parts:
            continue
        bad = {}
        for sp in glob.glob(os.path.join(cell, "score", model, "**", "BFCL_v4_%s_score.json" % part),
                            recursive=True):
            for i, rec in enumerate(rows(sp)):
                if (i or "total_count" not in rec) and rec.get("id"):
                    bad[rec["id"]] = rec
        info = {}
        dpath = os.path.join(data_dir, "BFCL_v4_%s.json" % part)
        if os.path.exists(dpath):
            info = {r["id"]: r for r in rows(dpath) if r.get("id")}
        gold = {}
        gpath = os.path.join(data_dir, "possible_answer", "BFCL_v4_%s.json" % part)
        if os.path.exists(gpath):
            gold = {r["id"]: r.get("ground_truth") for r in rows(gpath) if r.get("id")}

        out_dir = os.path.join(cell, "review", model)
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        with open(os.path.join(out_dir, "BFCL_v4_%s.jsonl" % part), "w") as fh:
            for rec in rows(rpath):
                rid = rec.get("id", "")
                wrong = bad.get(rid)
                src_rec = info.get(rid, {})
                item = {"reward": 0 if wrong else 1, "id": rid,
                        "query": query_of(src_rec),
                        "prompt": prompt_of(src_rec),
                        "response": rec.get("result", "")}
                if wrong:
                    item["error"] = wrong.get("error")
                    item["gold"] = gold.get(rid)
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        written[part] = n
    return written


def data_dir_for(data_root, variant, ptype):
    if ptype == "clean":
        return os.path.join(data_root, "clean", variant)
    return os.path.join(data_root, "pert_%s" % variant, ptype)


def review_tree(out_dir, model, data_root, parts=()):
    done = []
    for vdir in sorted(glob.glob(os.path.join(out_dir, "*"))):
        if not os.path.isdir(vdir):
            continue
        v = os.path.basename(vdir)
        for cdir in sorted(glob.glob(os.path.join(vdir, "*"))):
            t = os.path.basename(cdir)
            if not os.path.isdir(os.path.join(cdir, "result", model)):
                continue
            src = data_dir_for(data_root, v, t)
            done.append((v, t, review_cell(cdir, model, src, parts), src))
    return done


def discover_root(out, model):
    scratch = os.path.dirname(os.path.dirname(out))
    cands = sorted(glob.glob(os.path.join(scratch, "bfcl_data", "*")))
    cells = [(v, t) for v, t, _ in
             [(os.path.basename(os.path.dirname(c)), os.path.basename(c), c)
              for c in glob.glob(os.path.join(out, "*", "*")) if os.path.isdir(c)]
             if os.path.isdir(os.path.join(out, v, t, "result", model))]
    if not cells:
        return ""
    full = [c for c in cands
            if os.path.isdir(c) and all(os.path.isdir(data_dir_for(c, v, t)) for v, t in cells)]
    if len(full) == 1:
        print("data tree not recorded; using the one beside this run that covers all %d cells: %s"
              % (len(cells), full[0]))
        return full[0]
    if len(full) > 1:
        print("several data trees cover this run -- name one with --data-root:")
        for c in full:
            print("  %s" % c)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="a run's output tree: every cell in it is joined")
    ap.add_argument("--cell", default="", help="one cell instead: <out>/<variant>/<type> (needs --data)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", default="", help="the data tree the run used; read from <out>/.data_root "
                                                   "when the sweep wrote one")
    ap.add_argument("--data", default="", help="with --cell: the data dir THAT cell was run on")
    ap.add_argument("--parts", default="", help="comma list; default every partition present")
    a = ap.parse_args()
    parts = tuple(p for p in a.parts.split(",") if p)

    if a.cell:
        if not a.data:
            raise SystemExit("--cell needs --data: the data dir that cell was run on")
        got = review_cell(a.cell, a.model, a.data, parts)
        print("review: " + (", ".join("%s=%d" % kv for kv in sorted(got.items())) or "nothing to join"))
        return
    if not a.out:
        raise SystemExit("pass --out RUN_DIR (every cell) or --cell DIR --data DIR (one)")
    out = os.path.normpath(a.out)
    root = a.data_root
    if not root:
        marker = os.path.join(out, ".data_root")
        if os.path.exists(marker):
            root = open(marker).read().strip()
    if not root:
        root = discover_root(out, a.model)
    if not root or not os.path.isdir(root):
        raise SystemExit("no data tree for %s\n  The queries are in NEITHER result/ nor score/ -- they live "
                         "in the data files, and which data\n  a cell was run on is recorded nowhere else. "
                         "Pass --data-root DIR (the tree the run used:\n  $S/bfcl_data/<parts>). eval_open.sh records it in "
                         "<out>/.data_root, and then no flag is needed." % out)
    a.out = out
    rows_ = review_tree(a.out, a.model, root, parts)
    if not rows_:
        raise SystemExit("no cell under %s has result/%s -- is --model the tag the sweep used?"
                         % (a.out, a.model))
    print("data tree %s" % root)
    for v, t, got, src in rows_:
        n = sum(got.values())
        print("  %-8s %-20s %6d records  <- %s" % (v, t, n, os.path.relpath(src, root)))
    print("%d cells, %d records" % (len(rows_), sum(sum(g.values()) for _, _, g, _ in rows_)))


if __name__ == "__main__":
    main()
