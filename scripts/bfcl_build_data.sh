#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$TMPDIR"
PY=${PY:-python3}
BLD=${BLD:-$(dirname "$HERE")}
: "${SRCDATA:?set SRCDATA to a VERIFIED-pristine BFCL data dir (with possible_answer/)}"
: "${BFCL_SRC:?set BFCL_SRC to the berkeley-function-call-leaderboard dir (for bfcl_eval)}"
export PYTHONPATH="$BFCL_SRC${PYTHONPATH:+:$PYTHONPATH}"
OUT=${OUT:?set OUT to the data dir to build}
PARTS=${PARTS:-multiple,parallel,parallel_multiple}
DECOY_INSTR=${DECOY_INSTR:-default}
REWARD="cost_decoy cost_decoy_nt cost_decoy_abbrev latency_decoy latency_decoy_nt latency_decoy_abbrev"

BARE="$PARTS"
STEMS=""; FILES=""
for p in ${PARTS//,/ }; do
  STEMS="$STEMS,BFCL_v4_$p"
  FILES="$FILES BFCL_v4_$p.json"
done
STEMS="${STEMS#,}"; FILES="${FILES# }"
echo "=== partitions: $PARTS ==="

assert_files() {
  local d="$1" f
  for f in $FILES; do
    [ -s "$d/$f" ] || { echo "ASSERT FAIL: missing/empty $d/$f" >&2; exit 1; }
    [ -s "$d/possible_answer/$f" ] || { echo "ASSERT FAIL: missing gold $d/possible_answer/$f" >&2; exit 1; }
  done
}

echo "=== gate 1: is SRCDATA unperturbed? ==="
$PY - "$SRCDATA" "$FILES" <<'PY'
import json,sys,re
d,files=sys.argv[1],sys.argv[2].split()
TAG=re.compile(r'\[Cost: \$|\[Response time:|\[Latency:|costs approximately|typically responds in')
bad=[]
n=0
for f in files:
    for line in open(f"{d}/{f}"):
        line=line.strip()
        if not line: continue
        r=json.loads(line); n+=1
        names=[x.get("name") for x in r.get("function",[])]
        if len(names)!=len(set(names)): bad.append((f,r.get("id"),"duplicate tool name"))
        if any(TAG.search(x.get("description") or "") for x in r.get("function",[])):
            bad.append((f,r.get("id"),"cost/latency tag present"))
        if any(re.search(r"_1$",x or "") and (x or "")[:-2] in names for x in names):
            bad.append((f,r.get("id"),"duplicate twin X and X_1"))
        if any(re.fullmatch(r"function_\d+",x or "") for x in names): bad.append((f,r.get("id"),"already canonical"))
if bad:
    print("CONTAMINATED source, first 5:",bad[:5]); sys.exit(2)
print("source looks unperturbed over %d datapoints (no dup names / no tags / no _1 suffixes / not canonical)"%n)
PY

echo "=== gate 2: perturbation validity over every datapoint ==="
EXPECT=$($PY - "$SRCDATA" "$FILES" <<'PY'
import sys
d,files=sys.argv[1],sys.argv[2].split()
print(sum(1 for f in files for l in open(f"{d}/{f}") if l.strip()))
PY
)
echo "  expecting $EXPECT datapoints"
$PY $BLD/perturb/validate_perturb.py "$SRCDATA" --categories "$BARE" --examples 1 \
    --decoy-instruction "$DECOY_INSTR" | tee "$TMPDIR/perturb_validity.txt"
$PY - "$TMPDIR/perturb_validity.txt" "$EXPECT" "$DECOY_INSTR" <<'PY'
import re,sys
txt=open(sys.argv[1]).read(); expect=int(sys.argv[2]); style=sys.argv[3]
m=re.search(r'validity over (\d+) datapoints',txt)
if not m or int(m.group(1))!=expect:
    print("VALIDITY GATE FAILED: expected %d datapoints, saw %s"%(expect,m and m.group(1))); sys.exit(3)
rows=[l.split() for l in txt.splitlines() if re.match(r'^(query_typos|redundant|same_name_|cost_|latency_)',l)]
if len(rows)!=13:
    print("VALIDITY GATE FAILED: expected 13 type rows, saw",len(rows)); sys.exit(3)
COLS=["gold_declared","gold_args_ok","recoverable","distractor_distinct","non_inert",
      "clash_as_intended","variant_faithful","preference_determined","objective_unstated",
      "harness_accepts"]
bad=[]; gold_args=set()
for r in rows:
    name,cells=r[0],r[1:1+len(COLS)]
    for col,v in zip(COLS,cells):
        if col=="harness_accepts" and v=="n/a":
            print("VALIDITY GATE FAILED: harness_accepts not checked (bfcl_eval not importable)"); sys.exit(3)
        v=int(v)
        if col=="gold_args_ok":
            gold_args.add(v); continue
        if name=="query_typos" and col in ("recoverable","non_inert"): continue
        if col=="objective_unstated" and style=="inventory": continue
        if v: bad.append((name,col,v))
if bad:
    print("VALIDITY GATE FAILED:",bad); sys.exit(3)
if len(gold_args)!=1:
    print("VALIDITY GATE FAILED: gold_args_ok differs across types %s -- a perturbation is introducing "
          "its own gold/schema contradiction, not inheriting BFCL's"%sorted(gold_args)); sys.exit(3)
print("validity gate passed (%d gold/schema contradictions inherited from the clean data, identical on "
      "every type; accepted typo columns aside)"%gold_args.pop())
PY

mkdir -p "$OUT"
echo "=== clean: normal + canon ==="
rm -rf "$OUT/clean"; mkdir -p "$OUT/clean/normal/possible_answer"
for f in $FILES; do
  cp "$SRCDATA/$f" "$OUT/clean/normal/$f"
  cp "$SRCDATA/possible_answer/$f" "$OUT/clean/normal/possible_answer/$f"
done
assert_files "$OUT/clean/normal"
$PY $BLD/canonperm/canonicalize_bfcl.py "$OUT/clean/normal" "$OUT/clean/canon" --categories $STEMS
assert_files "$OUT/clean/canon"

echo "=== 13 types: normal ==="
rm -rf "$OUT/pert_normal"
$PY $BLD/perturb/build_bfcl_perturb.py "$OUT/clean/normal" "$OUT/pert_normal" --categories $STEMS \
    --decoy-instruction "$DECOY_INSTR"
for d in "$OUT"/pert_normal/*/; do assert_files "$d"; done

echo "=== 6 reward types: decoy-first (source for canon_sw) ==="
rm -rf "$OUT/pert_sw_real"
$PY $BLD/perturb/build_bfcl_perturb.py "$OUT/clean/normal" "$OUT/pert_sw_real" --categories $STEMS \
    --types "$(echo $REWARD | tr ' ' ',')" --decoy-first --decoy-instruction "$DECOY_INSTR"
for d in "$OUT"/pert_sw_real/*/; do assert_files "$d"; done

echo "=== canon for all 13, canon_sw for the 6 reward types ==="
rm -rf "$OUT/pert_canon" "$OUT/pert_canon_sw"
for d in "$OUT"/pert_normal/*/; do
  t=$(basename "$d")
  $PY $BLD/canonperm/canonicalize_bfcl.py "$d" "$OUT/pert_canon/$t" --categories $STEMS >/dev/null
  assert_files "$OUT/pert_canon/$t"
done
for d in "$OUT"/pert_sw_real/*/; do
  t=$(basename "$d")
  $PY $BLD/canonperm/canonicalize_bfcl.py "$d" "$OUT/pert_canon_sw/$t" --categories $STEMS >/dev/null
  assert_files "$OUT/pert_canon_sw/$t"
done

echo "=== gate 3: canon cells carry no real identifiers at ANY schema depth ==="
$PY - "$OUT" "$FILES" <<'PY'
import json,os,re,sys
out,FILES=sys.argv[1],sys.argv[2].split()
FN=re.compile(r'^function_\d\d$'); AR=re.compile(r'^arg_\d\d$')
ALLOW={("parallel_29","required",("adults","children","singles"))}
def leaks(d):
    bad=[]
    def walk(v,iid):
        if not isinstance(v,dict): return
        props=v.get("properties")
        if isinstance(props,dict):
            for k,sub in props.items():
                if not AR.match(k): bad.append("%s: arg key %r"%(iid,k))
                walk(sub,iid)
        for key in ("required","optional"):
            val=v.get(key)
            if isinstance(val,list):
                for x in val:
                    if isinstance(x,str) and not AR.match(x):
                        if (iid,key,tuple(val)) in ALLOW: break
                        bad.append("%s: %s %r"%(iid,key,x))
        it=v.get("items")
        if isinstance(it,dict): walk(it,iid)
        elif isinstance(it,list):
            for x in it: walk(x,iid)
    for f in FILES:
        p=os.path.join(d,f)
        if not os.path.exists(p): return ["missing %s"%f]
        for line in open(p):
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            for t in r.get("function",[]):
                if not FN.match(t.get("name") or ""):
                    bad.append("%s: tool name %r"%(r["id"],t.get("name")))
                walk(t.get("parameters") or {},r["id"])
    return bad
cells=[("clean/canon",os.path.join(out,"clean/canon"))]
for sub in ("pert_canon","pert_canon_sw"):
    base=os.path.join(out,sub)
    for t in sorted(os.listdir(base)): cells.append((sub+"/"+t,os.path.join(base,t)))
nbad=0
for name,d in cells:
    b=leaks(d)
    if b:
        nbad+=1; print("  LEAK %-30s %d: %s"%(name,len(b),sorted(set(b))[:3]))
if nbad: print("CANON GATE FAILED: %d/%d cells leak"%(nbad,len(cells))); sys.exit(4)
print("canon gate passed: all %d cells fully placeholder-labelled at every depth"%len(cells))
PY

printf '%s\n' "$DECOY_INSTR" > "$OUT/decoy_instruction.txt"

echo "=== gate 4: every built cell survives BFCL's own pre-prompt schema rewrite ==="
$PY $BLD/perturb/scan_harness_accepts.py "$OUT" "$PARTS"
echo "  normal=$(ls -1 $OUT/pert_normal | wc -l) canon=$(ls -1 $OUT/pert_canon | wc -l) canon_sw=$(ls -1 $OUT/pert_canon_sw | wc -l)"
echo BUILD_V4_OK
