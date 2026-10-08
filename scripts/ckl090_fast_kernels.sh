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
ENVNAME=${2:-${ENVNAME:-ckl090}}
if [ -z "$NVME" ]; then
  echo "usage: bash $0 SCRATCH_DIR [ENVNAME]" >&2
  echo "  SCRATCH_DIR is the data volume this box's env, HOME and caches live on (\`df -h\` shows what is" >&2
  echo "  mounted here). It is required, not defaulted: a wrong one fills the root volume." >&2
  exit 2
fi
[ -d "$NVME" ] && [ -w "$NVME" ] || { echo "FATAL: $NVME is not a writable directory" >&2; exit 1; }
case "$NVME" in /) echo "FATAL: refusing to use the root volume as scratch" >&2; exit 1 ;; esac

export HOME=$NVME/home TMPDIR=$NVME/tmp
export PIP_CACHE_DIR=$HOME/.cache/pip XDG_CACHE_HOME=$HOME/.cache HF_HOME=$HOME/.cache/hf
export TRITON_CACHE_DIR=$HOME/.cache/triton TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME"

CONDA=${CONDA:-$NVME/miniconda3/bin/conda}
[ -x "$CONDA" ] || { echo "FATAL: no conda at $CONDA (pass CONDA=/path/to/conda)" >&2; exit 1; }
BIN=$("$CONDA" info --base)/envs/$ENVNAME/bin
[ -x "$BIN/python" ] || { echo "FATAL: no env '$ENVNAME' at $BIN" >&2; exit 1; }

echo "=== before ==="
"$BIN/python" - <<'PY'
import importlib.metadata as md
for p in ("torch", "triton", "transformers", "vllm", "flash-attn", "einops",
          "flash-linear-attention", "causal-conv1d"):
    try:
        print("  %-24s %s" % (p, md.version(p)))
    except Exception:
        print("  %-24s MISSING" % p)
PY
TORCH_BEFORE=$("$BIN/python" -c "import torch;print(torch.__version__)")

export MAX_JOBS=${MAX_JOBS:-$(nproc)}
echo
echo "--- einops ---"
"$BIN/pip" install einops
echo
echo "--- fla-core + flash-linear-attention (Triton kernels, no compile) ---"
"$BIN/pip" install --no-deps fla-core flash-linear-attention
echo
echo "--- causal-conv1d (COMPILES CUDA, expect several minutes at MAX_JOBS=$MAX_JOBS) ---"
"$BIN/pip" install -v --no-deps --no-build-isolation causal-conv1d 2>&1 | grep -viE "^\s*(copying|creating|running|writing|reading|adding)" || {
    echo "causal-conv1d failed to build. Its nvcc output is above." >&2; exit 1; }

TORCH_AFTER=$("$BIN/python" -c "import torch;print(torch.__version__)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "FATAL: torch moved $TORCH_BEFORE -> $TORCH_AFTER. vllm and flash-attn were built against the old" >&2
  echo "  one, so serving in this env is now broken. Rebuild the env rather than repairing it." >&2
  exit 1
fi

echo
echo "=== the fast path, as transformers decides it ==="
"$BIN/python" - <<'PY'
import sys
ok = True
try:
    import triton
    v = tuple(int(x) for x in triton.__version__.split(".")[:2])
    print("  %s  triton %s (fla-core's cuda extra wants >= 3.3)"
          % ("PASS " if v >= (3, 3) else "FAIL ", triton.__version__))
    ok = ok and v >= (3, 3)
except Exception as e:
    ok = False; print("  FAIL  triton: %s: %s" % (type(e).__name__, e))
try:
    from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                          fused_recurrent_gated_delta_rule)
    print("  PASS  fla.ops.gated_delta_rule: chunk + fused_recurrent")
except Exception as e:
    ok = False; print("  FAIL  fla gated delta rule: %s: %s" % (type(e).__name__, e))
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print("  PASS  causal_conv1d_fn / causal_conv1d_update")
except Exception as e:
    ok = False; print("  FAIL  causal_conv1d: %s: %s" % (type(e).__name__, e))
for name in ("is_fla_available", "is_causal_conv1d_available"):
    try:
        import transformers.utils as U
        print("  transformers.%s() = %s" % (name, getattr(U, name)()))
    except Exception:
        pass
sys.exit(0 if ok else 3)
PY

echo
echo "READY. Relaunch training and confirm the warning is GONE:"
echo "  grep -c 'fast path is not available' <the new log>     # must be 0"
echo "Then read the step's own numbers:"
echo "  grep timing_s/step <log> | tail -1 | grep -oE 'timing_s/[a-z_]+:[0-9.]+|perf/mfu/actor:[0-9.]+'"
