#!/usr/bin/env bash
set -euo pipefail

KINDS=(); SCRATCH=""; CONDA_HOME=""; CONDA_ONLY=0; NO_SERVE=0
PY_TRAIN=3.10; PY_EVAL=3.10; PY_SERVE=3.11
BFCL_REPO=""; FORCE=0
_need() {
  case "${2-}" in
    "") echo "$1 requires a value, and none was given" >&2; exit 2 ;;
    -*) echo "$1 requires a value but got the option '$2' -- an empty shell variable?" >&2; exit 2 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --kind)      _need "$1" "${2-}"; KINDS+=("$2"); shift 2 ;;
    --scratch)   _need "$1" "${2-}"; SCRATCH="$2"; shift 2 ;;
    --conda)     _need "$1" "${2-}"; CONDA_HOME="$2"; shift 2 ;;
    --conda-only) CONDA_ONLY=1; shift ;;
    --bfcl-repo) _need "$1" "${2-}"; BFCL_REPO="$2"; shift 2 ;;
    --recreate)  FORCE=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ ${#KINDS[@]} -gt 0 ] || [ "$CONDA_ONLY" = 1 ] \
  || { echo "give at least one --kind train|eval|serve|tau, or --conda-only" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -z "$SCRATCH" ]; then
  echo "--scratch DIR is required: the data volume this run writes to (models, caches, HOME, outputs)." >&2
  echo "  It is a property of THIS machine, so it is never guessed. \`df -h\` shows what is mounted here." >&2
  exit 2
fi
[ -d "$SCRATCH" ] && [ -w "$SCRATCH" ] || { echo "--scratch $SCRATCH is not a writable directory" >&2; exit 1; }
case "$SCRATCH" in /) echo "refusing the root volume as scratch" >&2; exit 1 ;; esac
export HOME="$SCRATCH/home"
export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$HOME/.cache/pip" XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME="$HOME/.cache/hf" TRITON_CACHE_DIR="$HOME/.cache/triton"
export VLLM_CACHE_ROOT="$HOME/.cache/vllm" TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/inductor"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME"
echo "scratch  $SCRATCH   (HOME and every cache repointed here)"

if [ -z "$CONDA_HOME" ]; then
  CONDA_HOME="$SCRATCH/miniconda3"
  if [ ! -x "$CONDA_HOME/bin/conda" ]; then
    case "$(uname -s)-$(uname -m)" in
      Linux-x86_64)  U=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh ;;
      Linux-aarch64) U=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh ;;
      Darwin-arm64)  U=https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh ;;
      *) echo "no miniconda build known for $(uname -s)-$(uname -m); pass --conda PREFIX" >&2; exit 1 ;;
    esac
    echo "installing miniconda -> $CONDA_HOME"
    curl -fsSL "$U" -o "$TMPDIR/miniconda.sh"
    bash "$TMPDIR/miniconda.sh" -b -u -p "$CONDA_HOME"
  fi
  [ -x "$CONDA_HOME/bin/conda" ] || {
    echo "no runnable conda at $CONDA_HOME/bin/conda after install." >&2
    echo "  If $CONDA_HOME exists but is not a conda install, either point at the real one with" >&2
    echo "  --conda PREFIX, or move that directory aside and re-run. This script removes nothing." >&2
    exit 1; }
fi
"$CONDA_HOME/bin/conda" tos accept --override-channels --channel defaults >/dev/null 2>&1 || true
if [ "$CONDA_ONLY" = 1 ]; then
  echo "conda       $CONDA_HOME/bin/conda"
  echo "next        bash $HERE/scripts/ckl_verl090_env.sh --scratch $SCRATCH   (Qwen3.5 arm)"
  echo "            bash $HERE/setup_env.sh --scratch $SCRATCH --kind train    (Qwen2.5 arm)"
  exit 0
fi
CONDA="$CONDA_HOME/bin/conda"
echo "conda    $CONDA"

mk() {
  local name="$1" pyver="$2" prefix="$CONDA_HOME/envs/$1"
  if [ "$FORCE" = 1 ] && [ -d "$prefix" ]; then "$CONDA" env remove -y -n "$name" >/dev/null; fi
  [ -x "$prefix/bin/python" ] || "$CONDA" create -y -n "$name" "python=$pyver" >/dev/null
  echo "$prefix/bin"
}

for kind in "${KINDS[@]}"; do
case "$kind" in
train)
  BIN=$(mk rapt-train "$PY_TRAIN")
  bash "$HERE/scripts/apply_verl_patch.sh"
  TRAIN_SPEC=$HERE/env/rapt-train.txt
  [ -f "$TRAIN_SPEC" ] || { echo "FATAL: no pinned spec at $TRAIN_SPEC" >&2; exit 1; }
  "$BIN/pip" install -q --upgrade pip
  "$BIN/pip" install -q torch --index-url https://download.pytorch.org/whl/cu121 -c "$TRAIN_SPEC"
  grep -v '^[[:space:]]*flash-attn' "$HERE/toolrl/requirements.txt" > "$TMPDIR/toolrl-req.txt"
  "$BIN/pip" install -q -r "$TMPDIR/toolrl-req.txt" -c "$TRAIN_SPEC"
  "$BIN/pip" install -q flash-attn --no-build-isolation -c "$TRAIN_SPEC" || echo "  WARN: flash-attn failed; \
use_remove_padding=False or install it by hand"
  "$BIN/pip" install -q -e "$HERE/toolrl"
  "$BIN/pip" install -q huggingface_hub
  echo "train env  $BIN"
  ;;
eval)
  BIN=$(mk rapt-eval "$PY_EVAL")
  "$BIN/pip" install -q --upgrade pip
  EVAL_SPEC=$HERE/env/bfcl.txt
  if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    EVAL_SPEC=$TMPDIR/bfcl-nocuda.txt
    grep -vE '^[[:space:]]*(cuda-[A-Za-z0-9._-]+|nvidia-[A-Za-z0-9._-]+|triton)([=<>!~[:space:]]|$)' \
      "$HERE/env/bfcl.txt" > "$EVAL_SPEC"
    _drop=$(( $(wc -l < "$HERE/env/bfcl.txt") - $(wc -l < "$EVAL_SPEC") ))
    echo "  host is $(uname -s)/$(uname -m): dropped $_drop CUDA-only pin(s) from the eval spec"
    echo "  (cuda-*, nvidia-*, triton -- linux-x86_64 wheels; the harness does not import them)"
  fi
  "$BIN/pip" install -q -r "$EVAL_SPEC"
  BFCL_REPO=${BFCL_REPO:-$SCRATCH/bench/gorilla}
  BFCL_REF=${BFCL_REF:-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8}
  if [ ! -d "$BFCL_REPO/berkeley-function-call-leaderboard" ]; then
    mkdir -p "$(dirname "$BFCL_REPO")"
    git clone -q https://github.com/ShishirPatil/gorilla.git "$BFCL_REPO"
    git -C "$BFCL_REPO" checkout -q "$BFCL_REF"
  fi
  have=$(git -C "$BFCL_REPO" rev-parse HEAD 2>/dev/null || echo none)
  [ "$have" = "$BFCL_REF" ] || echo "  WARN: $BFCL_REPO is at $have, not the pinned $BFCL_REF"
  "$BIN/pip" install -q -e "$BFCL_REPO/berkeley-function-call-leaderboard"
  echo "eval env   $BIN"
  echo "bfcl repo  $BFCL_REPO/berkeley-function-call-leaderboard  ($BFCL_REF)"
  ;;
serve)
  BIN=$(mk rapt-serve "$PY_SERVE")
  "$BIN/pip" install -q --upgrade pip
  "$BIN/pip" install -q -r "$HERE/env/serve.txt"
  "$BIN/pip" install -q 'ninja==1.13.0'
  echo "serve env  $BIN"
  ;;
*) echo "unknown --kind $kind (train|eval|serve|tau)" >&2; exit 2 ;;
esac
done
echo
echo "done. Pass these to the run scripts:"
echo "  train.sh       --python $CONDA_HOME/envs/rapt-train/bin/python3"
echo "  eval_open.sh   --bfcl-bin $CONDA_HOME/envs/rapt-eval/bin --serve-bin $CONDA_HOME/envs/rapt-serve/bin"
echo "  scratch        --scratch $SCRATCH"
