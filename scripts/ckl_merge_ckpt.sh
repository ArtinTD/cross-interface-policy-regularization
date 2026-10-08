#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=$(dirname "$HERE")
CKPT=""; OUT=""; SCRATCH=""; ENVNAME=""; PYBIN=""
_need() {
  case "${2-}" in
    "") echo "$1 requires a value, and none was given" >&2; exit 2 ;;
    -*) echo "$1 requires a value but got the option '$2' -- an empty shell variable?" >&2; exit 2 ;;
  esac
}
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt)   _need "$1" "${2-}"; CKPT="$2"; shift 2 ;;
    --out)    _need "$1" "${2-}"; OUT="$2"; shift 2 ;;
    --scratch) _need "$1" "${2-}"; SCRATCH="$2"; shift 2 ;;
    --env)    _need "$1" "${2-}"; ENVNAME="$2"; shift 2 ;;
    --python) _need "$1" "${2-}"; PYBIN="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$CKPT" ] || { echo "--ckpt is required: a global_step_N directory, or its actor/ subdirectory" >&2; exit 2; }
[ -d "$CKPT" ] || { echo "no such checkpoint directory: $CKPT" >&2; exit 1; }

[ -n "$SCRATCH" ] || { echo "--scratch DIR is required: the data volume this writes to (HOME, caches, output)." >&2
                       echo "  It is a property of THIS machine, so it is never guessed. \`df -h\` shows it." >&2
                       exit 2; }
[ -d "$SCRATCH" ] && [ -w "$SCRATCH" ] || { echo "--scratch $SCRATCH is not a writable directory" >&2; exit 1; }
case "$SCRATCH" in /) echo "refusing to use the root volume as scratch" >&2; exit 1 ;; esac

export HOME=$SCRATCH/home
export TMPDIR=$SCRATCH/tmp
export PIP_CACHE_DIR=$HOME/.cache/pip
export XDG_CACHE_HOME=$HOME/.cache
export HF_HOME=$HOME/.cache/huggingface
export TRITON_CACHE_DIR=$HOME/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

if [ -z "$PYBIN" ]; then
  ENVNAME=${ENVNAME:-ckl090}
  PYBIN=$SCRATCH/miniconda3/envs/$ENVNAME/bin/python
fi
[ -x "$PYBIN" ] || { echo "no interpreter at $PYBIN (pass --env NAME or --python PATH)" >&2; exit 1; }

"$PYBIN" -c "import fla.modules" 2>/dev/null || {
  echo "this env cannot construct a Qwen3.5 model: \`import fla.modules\` fails." >&2
  echo "  transformers imports it while loading the modeling file, so the merge would read every shard and" >&2
  echo "  then fail at the save. Install the linear-attention libraries into this env:" >&2
  echo "    bash $CODE/scripts/ckl090_fast_kernels.sh $SCRATCH ${ENVNAME:-ckl090}" >&2
  exit 1; }

SRC=$CKPT
case "$(basename "$CKPT")" in
  actor) : ;;
  *) [ -d "$CKPT/actor" ] && SRC=$CKPT/actor ;;
esac
ls "$SRC"/model_world_size_*_rank_*.pt >/dev/null 2>&1 || {
  echo "no FSDP shards in $SRC" >&2
  echo "  expected model_world_size_<N>_rank_<r>.pt; found:" >&2
  ls -1 "$SRC" | sed 's/^/    /' >&2
  exit 1; }
[ -d "$SRC/huggingface" ] || { echo "no $SRC/huggingface -- the merger takes the config and tokenizer from it" >&2
                               exit 1; }

STEP=$(basename "$(dirname "$SRC")")
OUT=${OUT:-$SCRATCH/models/merged-$(basename "$(dirname "$(dirname "$SRC")")")-$STEP}
if ls "$OUT"/*.safetensors >/dev/null 2>&1; then
  echo "already merged: $OUT"
  echo "$OUT"
  exit 0
fi
mkdir -p "$(dirname "$OUT")"

echo "=== merging $SRC"
echo "    shards   $(ls -1 "$SRC"/model_world_size_*_rank_*.pt | wc -l)"
echo "    target   $OUT"
"$PYBIN" -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$OUT"

ls "$OUT"/*.safetensors >/dev/null 2>&1 || { echo "MERGE PRODUCED NO WEIGHTS in $OUT" >&2; exit 1; }
[ -f "$OUT/config.json" ] || { echo "no config.json in $OUT" >&2; exit 1; }
ls "$OUT"/tokenizer* >/dev/null 2>&1 || { echo "no tokenizer files in $OUT -- vLLM will fail to load it" >&2
                                          exit 1; }
echo "=== merged"
du -sh "$OUT"
echo "$OUT"
