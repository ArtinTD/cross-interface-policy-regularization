#!/usr/bin/env bash
set -euo pipefail

SCRATCH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --scratch) case "${2-}" in
                 ""|-*) echo "--scratch requires a directory" >&2; exit 2 ;;
               esac
               NVME="$2"; shift 2 ;;
    *) SCRATCH_ARGS+=("$1"); shift ;;
  esac
done
set -- "${SCRATCH_ARGS[@]+"${SCRATCH_ARGS[@]}"}"
NVME=${NVME:-${1:-}}
if [ -z "$NVME" ]; then
  echo "usage: bash $0 SCRATCH_DIR          (the data volume the env, HOME and every cache go under)" >&2
  echo "  it is required, not defaulted: the path differs per machine and a wrong one fills the root volume." >&2
  echo "  It is a property of THIS machine, so it is never guessed. \`df -h\` shows what is mounted here." >&2
  exit 2
fi
[ -d "$NVME" ] && [ -w "$NVME" ] || { echo "FATAL: $NVME is not a writable directory" >&2; exit 1; }
case "$NVME" in /) echo "FATAL: refusing to use the root volume as scratch" >&2; exit 1 ;; esac
ENVNAME=${ENVNAME:-ckl090}
VERL_TAG=${VERL_TAG:-v0.9.0}
CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VERL_DIR=${VERL_DIR:-$NVME/bench/verl-090}
SPEC=${SPEC:-$CODE/env/ckl090.txt}
CONDA=${CONDA:-$NVME/miniconda3/bin/conda}

export HOME=$NVME/home TMPDIR=$NVME/tmp
export PIP_CACHE_DIR=$HOME/.cache/pip XDG_CACHE_HOME=$HOME/.cache HF_HOME=$HOME/.cache/hf
export TRITON_CACHE_DIR=$HOME/.cache/triton VLLM_CACHE_ROOT=$HOME/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME"

[ -x "$CONDA" ] || { echo "FATAL: no conda at $CONDA (pass CONDA=/path/to/conda)" >&2; exit 1; }
PREFIX=$("$CONDA" info --base)/envs/$ENVNAME
BIN=$PREFIX/bin

echo "=============================================================="
echo "env      $ENVNAME  ->  $PREFIX"
echo "verl     $VERL_TAG -> $VERL_DIR"
echo "code     $CODE"
echo "versions $SPEC"
[ -f "$SPEC" ] || { echo "FATAL: no pinned spec at $SPEC" >&2; exit 1; }
echo "=============================================================="

if [ "${SKIP_INSTALL:-0}" != 1 ]; then
  if [ -d "$PREFIX" ]; then
    echo "REFUSING: the env '$ENVNAME' already exists." >&2
    echo "  Existing envs:" >&2; "$CONDA" env list | awk 'NF && $1 !~ /^#/ {print "    "$1}' >&2
    echo "  Pick an unused name:  ENVNAME=ckl090b bash $0" >&2
    echo "  Or re-check only:     SKIP_INSTALL=1 ENVNAME=$ENVNAME bash $0" >&2
    exit 1
  fi
  "$CONDA" create -y -n "$ENVNAME" python=3.12 >/dev/null
  echo "created $PREFIX"

  if [ ! -d "$VERL_DIR/.git" ]; then
    mkdir -p "$(dirname "$VERL_DIR")"
    git clone -q --depth 1 --branch "$VERL_TAG" https://github.com/volcengine/verl.git "$VERL_DIR"
  fi
  echo "verl at $(git -C "$VERL_DIR" describe --tags 2>/dev/null || echo '?')"

  "$BIN/pip" install -q --upgrade pip
  "$BIN/pip" install -q -r "$SPEC"
  "$BIN/pip" install -e "$VERL_DIR[vllm]"
  "$BIN/pip" install -q -r "$VERL_DIR/requirements.txt"
  _jobs=$(( $( (nproc 2>/dev/null || echo 12) ) / 3 )); [ "$_jobs" -ge 1 ] || _jobs=1
  export PATH="$BIN:$PATH"
  command -v ninja >/dev/null || echo "  WARN: no ninja on PATH; the CUDA extensions will build serially"
  echo "  building flash-attn with MAX_JOBS=$_jobs (ninja: $(command -v ninja || echo MISSING))"
  MAX_JOBS=$_jobs NVCC_THREADS=2 "$BIN/pip" install -q flash-attn==2.8.3.post1 --no-build-isolation || echo "  WARN: flash-attn failed; set use_remove_padding=False or build it by hand"
  "$BIN/pip" install -q --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2
  "$BIN/pip" install -q --no-deps --no-build-isolation causal-conv1d==1.7.0
fi

echo
echo "--- installed versions ---"
"$BIN/python" - <<'PYLIST'
import importlib.metadata as md
for p in ("torch", "triton", "vllm", "transformers", "tensordict", "verl", "ray", "flash-attn",
          "fla-core", "flash-linear-attention", "causal-conv1d", "einops", "ninja"):
    try:
        print("  %-24s %s" % (p, md.version(p)))
    except Exception:
        print("  %-24s MISSING" % p)
PYLIST

echo
echo "READY: $ENVNAME."
echo "  python  $BIN/python"
echo "          bash $CODE/train_ckl090.sh $NVME --env $ENVNAME --model <MODEL> --dry-run"
