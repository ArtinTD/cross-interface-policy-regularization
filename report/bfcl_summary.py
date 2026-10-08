#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import sys

ERR = "Error during inference"


def wilson(c, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def read_cell(score_dir, drop_errors):
    cor = tot = errs = 0
    for path in glob.glob(os.path.join(score_dir, "**", "BFCL_v4_*_score.json"), recursive=True):
        with open(path) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if i == 0:
                    if "total_count" not in rec:
                        break
                    cor += rec["correct_count"]
                    tot += rec["total_count"]
                elif ERR in json.dumps(rec.get("error", "")) or ERR in json.dumps(rec.get("model_result_raw", "")):
                    errs += 1
    return cor, (tot - errs if drop_errors else tot), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--drop-inference-errors", action="store_true",
                    help="subtract requests that never reached the model instead of scoring them wrong")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cells, variants, types = {}, [], []
    for vdir in sorted(glob.glob(os.path.join(a.out_dir, "*"))):
        if not os.path.isdir(vdir):
            continue
        v = os.path.basename(vdir)
        for tdir in sorted(glob.glob(os.path.join(vdir, "*"))):
            t = os.path.basename(tdir)
            fixed = os.path.isfile(os.path.join(tdir, "score", ".checker"))
            for sub in (["score"] if fixed else ["score_goldname", "score"]):
                sd = os.path.join(tdir, sub, a.model)
                if os.path.isdir(sd):
                    cor, n, errs = read_cell(sd, a.drop_inference_errors)
                    if n:
                        cells[(v, t)] = dict(acc=100.0 * cor / n, n=n, errors=errs,
                                             scoring=sub if fixed else sub + " (pre-fix tree)",
                                             lo=wilson(cor, n)[1], hi=wilson(cor, n)[2])
                        if v not in variants:
                            variants.append(v)
                        if t not in types:
                            types.append(t)
                    break
    if not cells:
        sys.exit("no scored cells under %s for model %r" % (a.out_dir, a.model))

    order = ["clean", "query_typos", "redundant"] + ["same_name_" + c for c in "ABCDE"] + [
        "cost_decoy", "cost_decoy_nt", "cost_decoy_abbrev",
        "latency_decoy", "latency_decoy_nt", "latency_decoy_abbrev"] + [
        "ibm_p_quest", "ibm_enrich", "ibm_enrich_fdesc", "ibm_enrich_pdesc"]
    types.sort(key=lambda t: (order.index(t) if t in order else len(order), t))

    print("=== %s : %s ===" % (a.model, a.out_dir))
    print("accuracy %% +-95%% Wilson (n) per cell%s" %
          ("; gateway-errored requests dropped" if a.drop_inference_errors else ""))
    print("%-22s %s" % ("perturbation", "  ".join("%-26s" % v for v in variants)))
    for t in types:
        row = []
        for v in variants:
            c = cells.get((v, t))
            row.append("%-26s" % ("--" if not c else "%5.1f +-%.1f (%d)%s"
                                  % (c["acc"], (c["hi"] - c["lo"]) / 2, c["n"],
                                     "" if c["scoring"] == "score" else " !")))
        print("%-22s %s" % (t, "  ".join(row)))
    stale = sorted({(v, t) for (v, t), c in cells.items() if c["scoring"] != "score"})
    if stale:
        print("\n! = this cell predates scripts/bfcl_fix_checker.sh (no score/.checker). Its single-tool\n"
              "    categories were scored against the FIRST declared tool, which rewards the decoy on a\n"
              "    decoy-first cell. Re-score it: the fix is unconditional and needs no regeneration.")
    errs = sum(c["errors"] for c in cells.values())
    if errs:
        print("\ngateway/inference errors across cells: %d%s" %
              (errs, "" if a.drop_inference_errors else "  (counted as wrong; pass --drop-inference-errors)"))
    if a.json:
        json.dump({"%s/%s" % k: v for k, v in cells.items()}, open(a.json, "w"), indent=1)
        print("wrote %s" % a.json)


if __name__ == "__main__":
    main()
