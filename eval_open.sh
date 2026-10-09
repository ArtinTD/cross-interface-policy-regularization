#!/usr/bin/env bash
set -euo pipefail

MODEL=""; TAG=""; SERVER=""; GPU=""; PORT=""; OUT=""; DATA=""; SRCDATA=""; BFCL_DIR=""
BFCL_BIN=""; SERVE_BIN=""; SCRATCH=""; PARTS=""; VARIANTS=""; TYPES="ALL"; THREADS=""; DUP_FIRST=0
SCORE_ONLY=0
DECOY_INSTR="default"
FC=0; TOOL_PARSER="qwen3_coder"
_need() {
  case "${2-}" in
    "") echo "$1 requires a value, and none was given" >&2; exit 2 ;;
    -*) echo "$1 requires a value but got the option '$2' -- an empty shell variable?" >&2; exit 2 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)     _need "$1" "${2-}"; MODEL="$2"; shift 2 ;;
    --tag)       _need "$1" "${2-}"; TAG="$2"; shift 2 ;;
    --server)    _need "$1" "${2-}"; SERVER="$2"; shift 2 ;;
    --gpu)       _need "$1" "${2-}"; GPU="$2"; shift 2 ;;
    --port)      _need "$1" "${2-}"; PORT="$2"; shift 2 ;;
    --out)       _need "$1" "${2-}"; OUT="$2"; shift 2 ;;
    --data)      _need "$1" "${2-}"; DATA="$2"; shift 2 ;;
    --src-data)  _need "$1" "${2-}"; SRCDATA="$2"; shift 2 ;;
    --bfcl-repo) _need "$1" "${2-}"; BFCL_DIR="$2"; shift 2 ;;
    --bfcl-bin)  _need "$1" "${2-}"; BFCL_BIN="$2"; shift 2 ;;
    --serve-bin) _need "$1" "${2-}"; SERVE_BIN="$2"; shift 2 ;;
    --scratch)   _need "$1" "${2-}"; SCRATCH="$2"; shift 2 ;;
    --parts)     _need "$1" "${2-}"; PARTS="$2"; shift 2 ;;
    --variants)  _need "$1" "${2-}"; VARIANTS="$2"; shift 2 ;;
    --types)     _need "$1" "${2-}"; TYPES="$2"; shift 2 ;;
    --threads)   _need "$1" "${2-}"; THREADS="$2"; shift 2 ;;
    --decoy-instruction) _need "$1" "${2-}"; DECOY_INSTR="$2"; shift 2 ;;
    --fc)        FC=1; shift ;;
    --tool-parser) _need "$1" "${2-}"; TOOL_PARSER="$2"; shift 2 ;;
    --score-only) SCORE_ONLY=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 2; }
TAG=${TAG:-$(basename "$MODEL" | tr '_' '-')}
case "$TAG" in *_*) echo "--tag must contain no underscore (got $TAG)" >&2; exit 2 ;; esac
case "$DECOY_INSTR" in
  default|budget|rbtc|inventory|inventory+budget) ;;
  *) echo "--decoy-instruction is default|budget|rbtc|inventory|inventory+budget; got '$DECOY_INSTR'" >&2
     exit 2 ;;
esac

if [ -z "$SCRATCH" ]; then
  echo "--scratch DIR is required: the data volume this run writes to (models, caches, HOME, outputs)." >&2
  echo "  It is a property of THIS machine, so it is never guessed. \`df -h\` shows what is mounted here." >&2
  exit 2
fi
[ -d "$SCRATCH" ] && [ -w "$SCRATCH" ] || { echo "--scratch $SCRATCH is not a writable directory" >&2; exit 1; }
case "$SCRATCH" in /) echo "refusing the root volume as scratch" >&2; exit 1 ;; esac

ALL_PARTS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,live_simple,live_multiple,live_parallel,live_parallel_multiple
case "${PARTS:-}" in
  ""|all) PARTS=$ALL_PARTS ;;
  *)
     for _p in $(echo "$PARTS" | tr ',' ' '); do
       case ",$ALL_PARTS," in
         *",$_p,"*) ;;
         *) echo "unknown partition '$_p'. The ten gold-bearing partitions are:" >&2
            echo "  $ALL_PARTS" >&2
            echo "Omit --parts for all of them (n=2501, the reported set)." >&2; exit 2 ;;
       esac
     done ;;
esac
export HOME="$SCRATCH/home" TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/home/.cache"
export HF_HOME="$SCRATCH/home/.cache/hf" TRITON_CACHE_DIR="$SCRATCH/home/.cache/triton"
export VLLM_CACHE_ROOT="$SCRATCH/home/.cache/vllm" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/home/.cache/inductor"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME"

pick() { for c in "$@"; do [ -x "$c" ] && { echo "$c"; return 0; }; done; return 0; }
pick_dir() { for c in "$@"; do [ -d "$c" ] && { echo "$c"; return 0; }; done; return 0; }
BFCL_BIN=${BFCL_BIN:-$(pick "$SCRATCH/miniconda3/envs/rapt-eval/bin/bfcl" "$(command -v bfcl || true)")}
[ -x "${BFCL_BIN:-}" ] || { echo "no bfcl at $SCRATCH/miniconda3/envs/rapt-eval/bin nor on PATH." >&2
  echo "  create it: bash $HERE/setup_env.sh --scratch $SCRATCH --kind eval" >&2
  echo "  or name one: --bfcl-bin /abs/path/to/bin" >&2; exit 1; }
EVAL_PY=$(dirname "$BFCL_BIN")/python
if [ -z "$SERVER" ]; then
  SERVE_BIN=${SERVE_BIN:-$(pick "$SCRATCH/miniconda3/envs/rapt-serve/bin/vllm" "$(command -v vllm || true)")}
  [ -x "${SERVE_BIN:-}" ] || { echo "no vllm at $SCRATCH/miniconda3/envs/rapt-serve/bin nor on PATH." >&2
    echo "  create it: bash $HERE/setup_env.sh --scratch $SCRATCH --kind serve" >&2
    echo "  or reuse a server: --server http://localhost:8300/v1" >&2; exit 1; }
fi

BASE_REPO=${BFCL_DIR:-$(pick_dir "$SCRATCH/bench/gorilla/berkeley-function-call-leaderboard" \
                                 "$HERE/../benchmarks/gorilla/berkeley-function-call-leaderboard")}
[ -d "${BASE_REPO:-}" ] || { echo "no BFCL repo; run setup_env.sh --kind eval, or pass --bfcl-repo" >&2; exit 1; }
INST=$SCRATCH/bfcl_installs/$TAG/berkeley-function-call-leaderboard
if [ ! -d "$INST/bfcl_eval/data" ]; then
  mkdir -p "$(dirname "$INST")"; cp -a "$BASE_REPO" "$INST"
  echo "cloned install -> $INST"
fi
BFCL_SRC="$INST" PY="$EVAL_PY" bash "$HERE/scripts/bfcl_fix_checker.sh"
REG_ARGS=(--handler rlla)
if [ "$FC" = 1 ]; then REG_ARGS=(--handler qwenfc --fc); fi
"$EVAL_PY" "$HERE/scripts/bfcl_register_model.py" --install "$INST" --name "$TAG" "${REG_ARGS[@]}" \
    --repo-root "$HERE"

case "$PARTS" in
  "$ALL_PARTS") _dname=all ;;
  *) _dname=$(echo "$PARTS" | tr ',' '+') ;;
esac
case "$DECOY_INSTR" in
  default) DATA=${DATA:-$SCRATCH/bfcl_data/$_dname} ;;
  *) DATA=${DATA:-$SCRATCH/bfcl_data/$_dname-$(echo "$DECOY_INSTR" | tr '+' '_')} ;;
esac
if [ -d "$DATA/clean/normal" ]; then
  _have=$( [ -f "$DATA/decoy_instruction.txt" ] && cat "$DATA/decoy_instruction.txt" || echo default )
  if [ "$_have" != "$DECOY_INSTR" ]; then
    echo "the data tree $DATA was built with --decoy-instruction $_have, but this run asks for" >&2
    echo "  $DECOY_INSTR. The reward families' query text differs between them, so its cells measure" >&2
    echo "  something else. Build that wording its own tree (omit --data and the name carries the" >&2
    echo "  wording), or run the wording the tree holds." >&2
    exit 2
  fi
fi
if [ ! -d "$DATA/clean/normal" ]; then
  SRCDATA=${SRCDATA:-$SCRATCH/bfcl_pristine}
  [ -d "$SRCDATA/possible_answer" ] || PY="$EVAL_PY" bash "$HERE/scripts/bfcl_seed_pristine.sh" "$SRCDATA"
  echo "building BFCL data -> $DATA"
  PY="$EVAL_PY" SRCDATA="$SRCDATA" BFCL_SRC="$INST" OUT="$DATA" PARTS="$PARTS" DUP_FIRST="$DUP_FIRST" \
      DECOY_INSTR="$DECOY_INSTR" bash "$HERE/scripts/bfcl_build_data.sh"
  printf '%s\n' "$PARTS" > "$DATA/parts.txt"
fi

if [ -n "$GPU" ]; then
  DEVS="$GPU"; NGPU=$(echo "$GPU" | tr ',' ' ' | wc -w | tr -d ' ')
else
  DEVS=""
  NGPU=$( { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true; } | wc -l | tr -d ' ')
fi
case "$NGPU" in ''|*[!0-9]*|0) NGPU=1 ;; esac
DP_FLAG=""; [ "$NGPU" -gt 1 ] && DP_FLAG="--data-parallel-size $NGPU"
true
THREADS=${THREADS:-$((32 * NGPU))}

SPID=""
stop_server() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null || true; }
trap stop_server EXIT INT TERM
_isuf=""
[ "$DECOY_INSTR" = default ] || _isuf="-$(echo "$DECOY_INSTR" | tr '+' '_')"
OUT=${OUT:-$SCRATCH/bfcl_out/$TAG$_isuf}
mkdir -p "$OUT"
if [ "$SCORE_ONLY" = 1 ]; then
  echo "--score-only: no model is served. Cells with responses are re-scored; a cell WITHOUT them is skipped"
  echo "  by the sweep's generate guard, so it stays missing rather than being generated."
  SERVER="${SERVER:-http://127.0.0.1:0/v1}"
elif [ -z "$SERVER" ]; then
  PORT=${PORT:-8300}
  LOG=$TMPDIR/serve_$TAG.log
  echo "serving $MODEL on ${DEVS:-all $NGPU GPUs} port $PORT, $NGPU replica(s), $THREADS threads  (log $LOG)"
  FC_FLAGS=""
  if [ "$FC" = 1 ]; then FC_FLAGS="--enable-auto-tool-choice --tool-call-parser $TOOL_PARSER"; fi
  env ${DEVS:+CUDA_VISIBLE_DEVICES="$DEVS"} "$SERVE_BIN" serve "$MODEL" --served-model-name "$TAG" \
      --port "$PORT" --dtype bfloat16 --gpu-memory-utilization 0.90 \
      $FC_FLAGS $DP_FLAG >"$LOG" 2>&1 &
  SPID=$!
  ok=0
  for _ in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 5
  done
  [ "$ok" = 1 ] || { echo "server never came up on :$PORT -- see $LOG" >&2; exit 1; }
  SERVER="http://localhost:$PORT/v1"
else
  if curl -sf --max-time 10 "$SERVER/models" >/dev/null 2>&1; then
    echo "using the server given: $SERVER"
  else
    echo "the server given does not answer: $SERVER" >&2
    echo "  tried GET $SERVER/models. A previous run's server is torn down when that run exits, so a URL" >&2
    echo "  that worked earlier is not alive now. Omit --server to have this script serve the model" >&2
    echo "  itself on every visible GPU." >&2
    exit 1
  fi
fi

env BFCL_BIN="$BFCL_BIN" BFCL_DIR="$INST" DATA_ROOT="$DATA" OUT="$OUT" \
  REMOTE_OPENAI_BASE_URL="$SERVER" TOKPATH="$MODEL" PARTS="$PARTS" THREADS="$THREADS" \
  MODEL="$TAG" ${VARIANTS:+VARIANTS="$VARIANTS"} TYPES="$TYPES" SCORE_ONLY="$SCORE_ONLY" \
  bash "$HERE/scripts/bfcl_sweep.sh"

echo
"$EVAL_PY" "$HERE/report/bfcl_summary.py" "$OUT" --model "$TAG"
echo "cells under $OUT"
