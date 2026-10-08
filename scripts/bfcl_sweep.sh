#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set MODEL to the served model tag}"
: "${BFCL_DIR:?set BFCL_DIR to the berkeley-function-call-leaderboard install to swap data into}"
: "${DATA_ROOT:?set DATA_ROOT to a dir built by scripts/bfcl_build_data.sh}"
: "${OUT:?set OUT to the output tree for this model}"
BFCL_BIN=${BFCL_BIN:-$(command -v bfcl || true)}
[ -x "$BFCL_BIN" ] || { echo "no bfcl binary; set BFCL_BIN to <eval env>/bin/bfcl" >&2; exit 1; }
PORT=${PORT:-8300}
THREADS=${THREADS:-32}
FIXMARK=$BFCL_DIR/bfcl_eval/eval_checker/ast_eval/.rapt_checker_fix
[ -s "$FIXMARK" ] || { echo "FATAL: $BFCL_DIR has not been fixed. Run:" >&2
  echo "         BFCL_SRC=$BFCL_DIR bash scripts/bfcl_fix_checker.sh" >&2; exit 1; }

HANDLERS="bfcl_eval/model_handler/local_inference/rlla_handler.py
bfcl_eval/model_handler/local_inference/qwen_fc_handler.py
bfcl_eval/model_handler/api_inference/claude.py
bfcl_eval/constants/model_config.py
bfcl_eval/constants/default_prompts.py"
_hsum=$( { for _h in $HANDLERS; do [ -f "$BFCL_DIR/$_h" ] && cat "$BFCL_DIR/$_h"; done \
           | { md5sum 2>/dev/null || md5 -q; } | awk '{print $1}'; } || true)
SCOREMARK="$(cat "$FIXMARK") handler=${_hsum:-none}"
export PYTHONPATH="$BFCL_DIR${PYTHONPATH:+:$PYTHONPATH}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORT_DIR=${REPORT_DIR:-$(dirname "$HERE")/report}
PY=${PY:-$(dirname "$BFCL_BIN")/python}
SRC=$BFCL_DIR/bfcl_eval/data
CLEAN=$DATA_ROOT/clean/normal
PARTS="${PARTS:-multiple,parallel,parallel_multiple}"
CATS="$PARTS"
FILES=""
for _p in ${PARTS//,/ }; do FILES="$FILES BFCL_v4_${_p}.json"; done
FILES="${FILES# }"
export REMOTE_OPENAI_BASE_URL="${REMOTE_OPENAI_BASE_URL:-http://localhost:${PORT}/v1}"
export REMOTE_OPENAI_API_KEY="${REMOTE_OPENAI_API_KEY:-EMPTY}"
export REMOTE_OPENAI_TOKENIZER_PATH="${TOKPATH:-$BFCL_DIR}"

restore() { local f; for f in $FILES; do
  cp -f "$CLEAN/$f" "$SRC/$f"; cp -f "$CLEAN/possible_answer/$f" "$SRC/possible_answer/$f"; done; }
trap restore EXIT INT TERM
swap() { local s="$1" f; for f in $FILES; do
  cp -f "$s/$f" "$SRC/$f"; cp -f "$s/possible_answer/$f" "$SRC/possible_answer/$f"; done; }

gate() {
  local s="$1" must="$2" f
  for f in $FILES; do
    [ -s "$s/$f" ] || { echo "GATE FAIL: missing $s/$f" >&2; return 1; }
    if [ "$must" = 1 ] && [ "$(md5sum <"$s/$f" | cut -d' ' -f1)" = "$(md5sum <"$CLEAN/$f" | cut -d' ' -f1)" ]; then
      echo "GATE FAIL: $s/$f is IDENTICAL to clean -- perturbation not applied" >&2; return 1
    fi
  done
}

have_responses() {
  [ -d "$1/result/$MODEL" ] || return 1
  "$PY" - "$1/result/$MODEL" "${2:-0}" <<'PYEOF'
import json, os, sys
want = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = err = 0
for dp, _, fs in os.walk(sys.argv[1]):
    for f in fs:
        if not f.endswith(".json"):
            continue
        for line in open(os.path.join(dp, f)):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            v = r.get("result")
            if isinstance(v, str) and v.startswith("Error during inference"):
                err += 1
print("%d %d %d" % (n, err, want))
sys.exit(0 if n and err < n and (not want or n >= want) else 1)
PYEOF
}

data_id() {
  local f
  for f in $FILES; do
    cat "$1/$f" "$1/possible_answer/$f" 2>/dev/null
  done | { md5sum 2>/dev/null || md5 -q; } | awk '{print $1}'
}

cell_records() {
  local f n=0 c
  for f in $FILES; do
    [ -s "$1/$f" ] || continue
    c=$(wc -l < "$1/$f")
    if [ -n "$(tail -c1 "$1/$f")" ]; then c=$((c + 1)); fi
    n=$((n + c))
  done
  echo "$n"
}

mkdir -p "$OUT"; printf '%s\n' "$DATA_ROOT" > "$OUT/.data_root"

review() {
  local o="$1" dir="$2"
  [ -d "$o/result/$MODEL" ] || return 0
  "$PY" "$REPORT_DIR/bfcl_review.py" --cell "$o" --model "$MODEL" --data "$dir" \
      --parts "$PARTS" 2>&1 | sed "s/^/[$CELL] /" || echo "[$CELL] review failed (numbers unaffected)" >&2
}

CELL=0
run_cell() {
  local v="$1" t="$2" dir="$3" must="$4"
  local o="$OUT/$v/$t"
  CELL=$((CELL + 1))
  local want was
  want=$(data_id "$dir")
  was=$( [ -f "$o/result/.data" ] && cat "$o/result/.data" || true )
  if [ -n "$was" ] && ! printf '%s' "$was" | grep -qE '^[0-9a-f]{32}$'; then
    if [ "$was" = "$(cell_records "$dir")" ]; then
      echo "[$CELL] $v/$t  legacy data marker ($was records) still matches this data; upgrading it"
      was="$want"
    fi
  fi
  if [ -n "$was" ] && [ "$was" != "$want" ]; then
    echo "[$CELL] $v/$t  REFUSED: these responses were generated against DIFFERENT data" >&2
    echo "    on disk $was" >&2
    echo "    now     $want   ($dir)" >&2
    echo "    Scoring them would compare old responses with new gold. Point --out at a fresh tree for this" >&2
    echo "    data (a non-default --decoy-instruction already does), or delete this cell to regenerate." >&2
    return 1
  fi
  local rowinfo have=1 rows
  rows=$(cell_records "$dir")
  rowinfo=$(have_responses "$o" "$rows") || have=0
  if [ "$have" = 0 ] && [ -d "$o/result/$MODEL" ]; then
    set -- $rowinfo
    if [ "${1:-0}" -lt "${3:-0}" ]; then
      echo "[$CELL] $v/$t  result/ is INCOMPLETE (${1:-0} of ${3:-0} records -- an interrupted generate)" >&2
      echo "               regenerating; otherwise evaluate fails on the length mismatch every time" >&2
    else
      echo "[$CELL] $v/$t  result/ holds only inference errors (${1:-0} row(s)) -- regenerating" >&2
    fi
  fi
  if [ "$have" = 1 ] && [ -f "$o/score/.checker" ] && [ -d "$o/score/$MODEL" ] \
     && [ "$(cat "$o/score/.checker")" = "$SCOREMARK" ]; then
    echo "[$CELL] [skip] $v/$t"
    review "$o" "$dir"
    return 0
  fi
  gate "$dir" "$must" || { echo "[$CELL] [GATED-OUT] $v/$t"; return 0; }
  mkdir -p "$o"; swap "$dir"
  local recs t0
  recs=$(cell_records "$dir"); t0=$(date +%s)
  if [ "$have" = 1 ]; then
    echo "[$CELL] $v/$t  re-scoring $recs records (responses already on disk, no model)"
  else
    echo "[$CELL] $v/$t  generating $recs records, $THREADS threads"
  fi
  if [ "$have" = 0 ]; then
    if [ "${SCORE_ONLY:-0}" = 1 ]; then
      echo "[$CELL] $v/$t  [skip] no responses on disk, and --score-only does not generate"
      return 0
    fi
    "$BFCL_BIN" generate --model "$MODEL" --test-category $CATS --skip-server-setup \
        --temperature 0.0 --num-threads "$THREADS" --result-dir "$o/result" -o \
      || { echo "[$CELL] $v/$t  GENERATE FAILED" >&2; return 1; }
  fi
  echo "[$CELL] $v/$t  scoring"
  "$BFCL_BIN" evaluate --model "$MODEL" --test-category $CATS \
      --result-dir "$o/result" --score-dir "$o/score" \
    || { echo "[$CELL] $v/$t  EVALUATE FAILED" >&2; return 1; }
  printf '%s\n' "$SCOREMARK" > "$o/score/.checker"
  printf '%s\n' "$want" > "$o/result/.data"
  review "$o" "$dir"
  echo "[$CELL] [done] $v/$t in $(( $(date +%s) - t0 ))s"
}

FAILED=""
VARIANTS="${VARIANTS:-normal canon canon_sw}"
TYPES="${TYPES:-ALL}"
for v in $VARIANTS; do
  case "$v" in
    normal)         base="$DATA_ROOT/pert_normal";         cleandir="$DATA_ROOT/clean/normal" ;;
    canon)          base="$DATA_ROOT/pert_canon";          cleandir="$DATA_ROOT/clean/canon" ;;
    canon_sw)       base="$DATA_ROOT/pert_canon_sw";       cleandir="" ;;
    *) echo "unknown variant $v" >&2; exit 1 ;;
  esac
  if [ -n "$cleandir" ] && { [ "$TYPES" = ALL ] || echo "$TYPES" | grep -qw clean; }; then
    run_cell "$v" clean "$cleandir" 0 || FAILED="$FAILED $v/clean"
  fi
  [ -d "$base" ] || continue
  for d in "$base"/*/; do
    t=$(basename "$d")
    if [ "$TYPES" != ALL ] && ! echo "$TYPES" | grep -qw "$t"; then continue; fi
    run_cell "$v" "$t" "$d" 1 || FAILED="$FAILED $v/$t"
  done
done
if [ -n "$FAILED" ]; then
  echo "CELLS THAT FAILED:$FAILED" >&2
  exit 1
fi
echo BFCL_SWEEP_COMPLETE
