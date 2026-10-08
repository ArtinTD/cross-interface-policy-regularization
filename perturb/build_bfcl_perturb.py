import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, "canonperm"))
import rbtc16 as R
from canonicalize_bfcl import load_records, is_single_turn_gold


def build_file(src_dir, out_dir, fname, ptype, donors, decoy_first=False, dup_first=False,
               instruction="default"):
    records = load_records(os.path.join(src_dir, fname))
    gold_by_id = {g["id"]: g.get("ground_truth", []) for g in load_records(os.path.join(src_dir, "possible_answer", fname))}
    sample = next((gold_by_id[r["id"]] for r in records if r["id"] in gold_by_id), [])
    if gold_by_id and not is_single_turn_gold(sample):
        return ("skip_multi_turn", len(records))
    out_records, out_gold = [], []
    for r in records:
        f2, q2, g2 = R.perturb(ptype, r.get("function", []), r.get("question", []),
                               gold_by_id.get(r["id"], []), r["id"], donors=donors,
                               decoy_first=decoy_first, dup_first=dup_first,
                               instruction=instruction)
        nr = dict(r)
        injected = [i for i, t in enumerate(f2) if isinstance(t, dict) and t.pop(R.INJECTED_KEY, None)]
        nr["function"] = f2
        nr["question"] = q2
        if injected:
            nr[R.INJECTED_KEY] = injected
        out_records.append(nr)
        if r["id"] in gold_by_id:
            out_gold.append({"id": r["id"], "ground_truth": g2})
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "possible_answer"), exist_ok=True)
    with open(os.path.join(out_dir, fname), "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if out_gold:
        with open(os.path.join(out_dir, "possible_answer", fname), "w") as f:
            for g in out_gold:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return ("ok", len(out_records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_data_dir")
    ap.add_argument("out_root")
    ap.add_argument("--types", default="", help="comma-separated rbtc16 type codes; default = all 13")
    ap.add_argument("--categories", default="", help="comma-separated file stems; default = all single-turn BFCL_*.json")
    ap.add_argument("--decoy-first", action="store_true",
                    help="reward types only: emit the decoy ahead of the renamed gold, so positional "
                         "canonicalization gives the decoy the lower function number (the canon_sw condition)")
    ap.add_argument("--dup-first", action="store_true",
                    help="duplicate types only (redundant, same_name_*): declare the injected tool BEFORE "
                         "the one it shadows. BFCL resolves a called name to the FIRST declared tool of "
                         "that name, so for the same-name family this changes which tool the harness "
                         "validates the gold call against; run validate_perturb on the result before "
                         "scoring it")
    ap.add_argument("--decoy-instruction", default="default", choices=list(R.INSTRUCTION_STYLES),
                    help="reward types only: which wording is appended to the query. `default` keeps every "
                         "reported number; `inventory` asks for an inventory of the menu and states no "
                         "objective, so that cell measures unprompted preference (see rbtc16)")
    args = ap.parse_args()
    types = [t.strip() for t in args.types.split(",") if t.strip()] or R.TYPES
    if args.categories:
        files = []
        for s in args.categories.split(","):
            s = s.strip()
            cand = [x for x in (f"{s}.json", f"BFCL_v4_{s}.json")
                    if os.path.exists(os.path.join(args.src_data_dir, x))]
            if not cand:
                print(f"UNRESOLVED CATEGORY {s}")
                continue
            files.append(cand[0])
    else:
        files = sorted(x for x in os.listdir(args.src_data_dir) if x.endswith(".json") and x.startswith("BFCL"))
    pools = {}
    for fname in files:
        lang = R.language_of(fname)
        if lang not in pools:
            pools[lang] = R.build_donor_pool(args.src_data_dir, lang)
            print(f"[donor pool] {lang}: {len(pools[lang])} distinct real BFCL tools")
    for ptype in types:
        if ptype not in R.TYPES:
            print(f"UNKNOWN TYPE {ptype}"); continue
        n_ok = 0
        for fname in files:
            if not os.path.exists(os.path.join(args.src_data_dir, fname)):
                continue
            try:
                status, n = build_file(args.src_data_dir, os.path.join(args.out_root, ptype), fname, ptype,
                                       pools[R.language_of(fname)], args.decoy_first, args.dup_first,
                                       instruction=args.decoy_instruction)
            except Exception as e:
                print(f"  ERROR {ptype}/{fname}: {type(e).__name__}: {str(e)[:70]}"); continue
            if status == "ok":
                n_ok += n
        print(f"{ptype:30s} {DISPLAY_NAME(ptype):32s} {n_ok} records")


def DISPLAY_NAME(t):
    return R.DISPLAY.get(t, t)


if __name__ == "__main__":
    main()
