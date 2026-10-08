#!/usr/bin/env bash
set -u
PY=${PY:-python3}
DEST=${1:?usage: seed_bfcl_pristine.sh DEST_DIR}
BFCL_REF=${BFCL_REF:-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8}
RAW=https://raw.githubusercontent.com/ShishirPatil/gorilla/$BFCL_REF/berkeley-function-call-leaderboard/bfcl_eval/data

PARTS="simple_python simple_java simple_javascript multiple parallel parallel_multiple irrelevance
       live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance"

mkdir -p "$DEST/possible_answer"
: > "$DEST/MD5SUMS"
fail=0
for c in $PARTS; do
  f=BFCL_v4_$c.json
  curl -sf -m 180 -o "$DEST/$f" "$RAW/$f" || { echo "FETCH-FAIL $f"; fail=1; continue; }
  n=$(grep -c . "$DEST/$f")
  if curl -sf -m 180 -o "$DEST/possible_answer/$f" "$RAW/possible_answer/$f"; then g=$(grep -c . "$DEST/possible_answer/$f"); else rm -f "$DEST/possible_answer/$f"; g="none"; fi
  printf "%-26s records=%-6s gold=%-6s\n" "$c" "$n" "$g"
  md5sum "$DEST/$f" >> "$DEST/MD5SUMS"
  [ -f "$DEST/possible_answer/$f" ] && md5sum "$DEST/possible_answer/$f" >> "$DEST/MD5SUMS"
done

echo "=== duplicate-named-tool check (must be 0 everywhere) ==="
"$PY" - "$DEST" <<'PY'
import json, os, sys
d = sys.argv[1]
bad = 0
for fn in sorted(x for x in os.listdir(d) if x.endswith(".json")):
    rs = [json.loads(l) for l in open(os.path.join(d, fn)) if l.strip()]
    n = 0
    for r in rs:
        fs = r.get("function")
        fs = fs if isinstance(fs, list) else [fs]
        names = [f.get("name", "") for f in fs if isinstance(f, dict)]
        if len(names) > len(set(names)):
            n += 1
    if n:
        print("  CONTAMINATED %s: %d/%d records" % (fn, n, len(rs)))
        bad += 1
print("  clean" if not bad else "  %d partition(s) still dirty" % bad)
sys.exit(1 if bad else 0)
PY
st=$?
chmod -R a-w "$DEST"
echo "sealed read-only: $DEST"
[ $fail = 0 ] && [ $st = 0 ] && echo "BFCL_PRISTINE_OK" || echo "BFCL_PRISTINE_PROBLEM"
