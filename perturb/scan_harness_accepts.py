import collections
import copy
import json
import os
import sys

from bfcl_eval.utils import _func_doc_language_specific_pre_processing as pre
from bfcl_eval.utils import extract_test_category_from_id as category_of


def cells(root):
    out = []
    for sub in ("clean/normal", "clean/canon"):
        if os.path.isdir(os.path.join(root, sub)):
            out.append((sub, os.path.join(root, sub)))
    for sub in ("pert_normal", "pert_canon", "pert_canon_sw"):
        base = os.path.join(root, sub)
        if os.path.isdir(base):
            for t in sorted(os.listdir(base)):
                out.append(("%s/%s" % (sub, t), os.path.join(base, t)))
    return out


def main():
    root, parts = sys.argv[1], sys.argv[2].split(",")
    sigs = collections.Counter()
    n_pairs = 0
    for name, d in cells(root):
        for p in parts:
            path = os.path.join(d, "BFCL_v4_%s.json" % p)
            if not os.path.exists(path):
                continue
            n_bad, first = 0, None
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                try:
                    pre(copy.deepcopy(r.get("function") or []), category_of(r["id"]))
                except Exception as e:
                    sig = "%s: %s" % (type(e).__name__, str(e)[:40])
                    n_bad += 1
                    sigs[sig] += 1
                    if first is None:
                        first = (r["id"], sig)
            if n_bad:
                n_pairs += 1
                print("REFUSED %-34s %-24s %4d records  first=%s %s"
                      % (name, p, n_bad, first[0], first[1]))
    if not n_pairs:
        print("harness gate passed: every cell survives BFCL's own pre-prompt schema rewrite")
        return 0
    print("\n%d (cell,partition) pairs refused by the harness" % n_pairs)
    for s, c in sigs.most_common():
        print("  %5d  %s" % (c, s))
    return 5


if __name__ == "__main__":
    sys.exit(main())
