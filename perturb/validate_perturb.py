import argparse
import collections
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import rbtc16 as R

try:
    from bfcl_eval.utils import _func_doc_language_specific_pre_processing as _bfcl_pre
    from bfcl_eval.utils import extract_test_category_from_id as _bfcl_cat
except Exception:
    _bfcl_pre = _bfcl_cat = None

CATS = ["multiple", "parallel", "parallel_multiple"]
INVARIANTS = ["gold_declared", "gold_args_ok", "recoverable", "distractor_distinct", "non_inert",
              "clash_as_intended", "variant_faithful", "preference_determined", "objective_unstated",
              "harness_accepts"]


def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["id"]] = r
    return out


def fn_sig(t):
    return json.dumps({"d": t.get("description"), "p": t.get("parameters")}, sort_keys=True)


def first_named(functions, name):
    for t in functions:
        if t.get("name") == name:
            return t
    return None


def gold_calls(gold):
    out = []
    for call in gold or []:
        if isinstance(call, dict):
            for fn, args in call.items():
                out.append((fn, args if isinstance(args, dict) else {}))
    return out


def gold_strings(gold, minlen=4):
    vals = set()

    def walk(v):
        if isinstance(v, str):
            vals.add(v)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    for _fn, args in gold_calls(gold):
        walk(args)
    return {v for v in vals if len(v) >= minlen}


def question_text(question):
    parts = []
    for turn in question or []:
        for m in turn if isinstance(turn, list) else []:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
    return "\n".join(parts)


def check(ptype, clean_fns, clean_q, fns, q, gold_clean, gold_pert, iid=""):
    bad = {}
    names = [t.get("name") for t in fns]
    gnames = [fn for fn, _a in gold_calls(gold_pert)]

    missing = [n for n in gnames if first_named(fns, n) is None]
    if missing:
        bad["gold_declared"] = "gold names not on menu: %s" % sorted(set(missing))[:3]

    reasons = []
    for fn, args in gold_calls(gold_pert):
        t = first_named(fns, fn)
        if t is None:
            continue
        props = (t.get("parameters") or {}).get("properties") or {}
        req = (t.get("parameters") or {}).get("required") or []
        unknown = [k for k in args if k not in props]
        absent = [k for k in req if k not in args]
        if unknown:
            reasons.append("%s: args not in schema %s" % (fn, sorted(unknown)[:3]))
        if absent:
            reasons.append("%s: required missing from gold %s" % (fn, sorted(absent)[:3]))
    if reasons:
        bad["gold_args_ok"] = "; ".join(reasons[:2])

    qc, qp = question_text(clean_q), question_text(q)
    lost = [v for v in gold_strings(gold_pert) if v in qc and v not in qp]
    if lost:
        bad["recoverable"] = "gold values erased from question: %s" % sorted(lost)[:3]

    gsigs = {}
    for fn, _a in gold_calls(gold_pert):
        t = first_named(fns, fn)
        if t is not None:
            gsigs[fn_sig(t)] = fn
    clones = [t.get("name") for t in fns
              if t.get("name") not in gnames and fn_sig(t) in gsigs]
    if clones:
        bad["distractor_distinct"] = "equally-correct clone(s) of gold: %s" % sorted(set(clones))[:3]

    if json.dumps(fns, sort_keys=True) == json.dumps(clean_fns, sort_keys=True) and qp == qc:
        bad["non_inert"] = "interface and question both unchanged"

    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    if ptype in R._SAME_NAME:
        need = [n for n in gnames if collections.Counter(names)[n] < 2]
        if need:
            bad["clash_as_intended"] = "no same-name twin for %s" % sorted(set(need))[:3]
    elif dup:
        bad["clash_as_intended"] = "unintended duplicate name(s): %s" % sorted(dup)[:3]

    if ptype in R._SAME_NAME:
        bad.update(_check_twin(ptype, clean_fns, fns))
    if ptype in R._REWARD:
        bad.update(_check_reward(ptype, fns, gold_pert, qp))
    if _bfcl_pre is not None and iid:
        try:
            _bfcl_pre(copy.deepcopy(fns), _bfcl_cat(iid))
        except Exception as e:
            bad["harness_accepts"] = "%s in BFCL pre-processing: %s" % (type(e).__name__, str(e)[:60])
    return bad


def _params_of(t):
    return json.dumps((t or {}).get("parameters"), sort_keys=True)


def _twins_of(clean_fns, fns):
    out = []
    for name in {t.get("name") for t in fns}:
        base = first_named(clean_fns, name)
        if base is None:
            continue
        same = [t for t in fns if t.get("name") == name]
        if len(same) < 2:
            continue
        marked = [t for t in same if t.get(R.INJECTED_KEY)]
        if marked:
            out.append((base, marked[0]))
            continue
        altered = [t for t in same if fn_sig(t) != fn_sig(base)]
        out.append((base, altered[0] if altered else same[-1]))
    return out


def _check_twin(ptype, clean_fns, fns):
    empty = json.dumps(R._empty_params(), sort_keys=True)
    reasons = []
    for base, twin in _twins_of(clean_fns, fns):
        same_p = _params_of(twin) == _params_of(base)
        same_d = twin.get("description") == base.get("description")
        if ptype == "same_name_A":
            if twin.get("description") or _params_of(twin) != empty:
                reasons.append("A twin should be blank desc + empty params")
        elif ptype == "same_name_B":
            if not same_d or _params_of(twin) != empty:
                reasons.append("B twin should keep desc + empty params")
        elif ptype in ("same_name_C", "same_name_D") and same_p:
            reasons.append("%s twin has the ORIGINAL schema, not a wrong one" % ptype[-1])
        elif ptype == "same_name_E" and (same_p or same_d):
            reasons.append("E twin should differ in BOTH desc and schema")
    if reasons:
        return {"variant_faithful": "; ".join(sorted(set(reasons))[:2])}
    return {}


def _check_reward(ptype, fns, gold_pert, qtext):
    gnames = {fn for fn, _a in gold_calls(gold_pert)}
    instr = R.states_preference(qtext)
    objective = R.states_objective(qtext)
    for t in fns:
        if t.get("name") in gnames:
            continue
        for gn in gnames:
            g = first_named(fns, gn)
            if g is None or _params_of(t) != _params_of(g):
                continue
            if not instr:
                return {"preference_determined":
                        "decoy %r matches gold schema but the query carries no injected instruction at all"
                        % t.get("name")}
            if not objective:
                return {"objective_unstated":
                        "decoy %r matches gold schema and the query states no objective, so the gold is not "
                        "determined by the query (expected under --decoy-instruction inventory)"
                        % t.get("name")}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_data_dir")
    ap.add_argument("--types", default="")
    ap.add_argument("--categories", default="")
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--dup-first", action="store_true",
                    help="validate the duplicate families with the injected tool declared FIRST")
    ap.add_argument("--decoy-first", action="store_true",
                    help="validate the reward families with the decoy declared ahead of the renamed gold, "
                         "which is the canon_sw condition that ships in the table and had no way to be "
                         "checked here before")
    ap.add_argument("--decoy-instruction", default="default", choices=list(R.INSTRUCTION_STYLES),
                    help="the wording the reward families append to the query; must match the build being "
                         "validated, or this checks data that was never written")
    a = ap.parse_args()

    types = [t.strip() for t in a.types.split(",") if t.strip()] or list(R.TYPES)
    cats = [c.strip() for c in a.categories.split(",") if c.strip()] or CATS

    data, gold = {}, {}
    for c in cats:
        d = load(os.path.join(a.src_data_dir, "BFCL_v4_%s.json" % c))
        g = load(os.path.join(a.src_data_dir, "possible_answer", "BFCL_v4_%s.json" % c))
        data.update(d)
        gold.update({k: v.get("ground_truth", []) for k, v in g.items()})
    n_items = len(data)

    counts = {t: collections.Counter() for t in types}
    examples = collections.defaultdict(list)
    pools = {lang: R.build_donor_pool(a.src_data_dir, lang)
             for lang in sorted({R.language_of(iid) for iid in data})}
    for ptype in types:
        for iid, rec in data.items():
            g0 = gold.get(iid, [])
            fns, q, g1 = R.perturb(ptype, rec.get("function", []), rec.get("question", []), g0, iid,
                                   donors=pools[R.language_of(iid)], dup_first=a.dup_first,
                                   decoy_first=a.decoy_first, instruction=a.decoy_instruction)
            bad = check(ptype, rec.get("function", []), rec.get("question", []), fns, q, g0, g1, iid=iid)
            for k, why in bad.items():
                counts[ptype][k] += 1
                if len(examples[(ptype, k)]) < a.examples:
                    examples[(ptype, k)].append((iid, why))

    hdr = "%-22s" % "type" + "".join("%14s" % k[:13] for k in INVARIANTS) + "%10s" % "ANY"
    print("BFCL perturbation validity over %d datapoints (%s)" % (n_items, ", ".join(cats)))
    print("donor pools: %s" % ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(pools.items())))
    if _bfcl_pre is None:
        print("harness_accepts NOT CHECKED: bfcl_eval is not importable, so V9 reads n/a, not 0")
    print("violations per invariant; 0 = the type satisfies it on every datapoint\n")
    print(hdr)
    print("-" * len(hdr))
    for ptype in types:
        row = "%-22s" % ptype + "".join(
            "%14s" % ("n/a" if k == "harness_accepts" and _bfcl_pre is None else counts[ptype][k])
            for k in INVARIANTS)
        worst = max(counts[ptype].values()) if counts[ptype] else 0
        print(row + "%10s" % ("" if not worst else "<=%d" % worst))
    print("\nexamples")
    for ptype in types:
        for k in INVARIANTS:
            for iid, why in examples[(ptype, k)]:
                print("  %-22s %-20s %-18s %s" % (ptype, k, iid, why))


if __name__ == "__main__":
    main()
