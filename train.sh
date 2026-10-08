#!/usr/bin/env bash
set -euo pipefail

PERT=twin; DRADD=on; CKL=rel+add
MODEL=""; DATA=""; OUTDIR=""; NAME=""; RESUME=""; RLLA_JSON=""; VERL=""; PYBIN=""; SCRATCH=""
ENVNAME=""
EPOCHS=""; GPUS=""; LAM_REL=""; LAM_ADD=""; DRY=0
_need() {
  case "${2-}" in
    "") echo "$1 requires a value, and none was given" >&2; exit 2 ;;
    -*) echo "$1 requires a value but got the option '$2' -- an empty shell variable?" >&2; exit 2 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)        _need "$1" "${2-}"; MODEL="$2"; shift 2 ;;
    --data)         _need "$1" "${2-}"; DATA="$2"; shift 2 ;;
    --rlla-json)    _need "$1" "${2-}"; RLLA_JSON="$2"; shift 2 ;;
    --out)          _need "$1" "${2-}"; OUTDIR="$2"; shift 2 ;;
    --name)         _need "$1" "${2-}"; NAME="$2"; shift 2 ;;
    --verl)         _need "$1" "${2-}"; VERL="$2"; shift 2 ;;
    --python)       _need "$1" "${2-}"; PYBIN="$2"; shift 2 ;;
    --env)          _need "$1" "${2-}"; ENVNAME="$2"; shift 2 ;;
    --scratch)      _need "$1" "${2-}"; SCRATCH="$2"; shift 2 ;;
    --epochs)       _need "$1" "${2-}"; EPOCHS="$2"; shift 2 ;;
    --gpus)         _need "$1" "${2-}"; GPUS="$2"; shift 2 ;;
    --lam-rel)      _need "$1" "${2-}"; LAM_REL="$2"; shift 2 ;;
    --lam-add)      _need "$1" "${2-}"; LAM_ADD="$2"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -z "$SCRATCH" ]; then
  echo "--scratch DIR is required: the data volume this run writes to (models, caches, HOME, outputs)." >&2
  echo "  It is a property of THIS machine, so it is never guessed. \`df -h\` shows what is mounted here." >&2
  exit 2
fi
[ -d "$SCRATCH" ] && [ -w "$SCRATCH" ] || { echo "--scratch $SCRATCH is not a writable directory" >&2; exit 1; }
case "$SCRATCH" in /) echo "refusing the root volume as scratch" >&2; exit 1 ;; esac
export HOME="$SCRATCH/home" TMPDIR="$SCRATCH/tmp" RAY_TMPDIR="$SCRATCH/tmp"
export XDG_CACHE_HOME="$HOME/.cache" PIP_CACHE_DIR="$HOME/.cache/pip" HF_HOME="$HOME/.cache/hf"
export TRITON_CACHE_DIR="$HOME/.cache/triton" VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/inductor"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME"
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

VERL=${VERL:-$HERE/toolrl}
export PYTHONPATH="$VERL${PYTHONPATH:+:$PYTHONPATH}"
if [ "$VERL" = "$HERE/toolrl" ] && ! bash "$HERE/scripts/apply_verl_patch.sh" --check >/dev/null 2>&1; then
  echo "the verl overlay is not applied to $VERL -- this run would be stock GRPO, not CKL." >&2
  echo "  fix: bash scripts/apply_verl_patch.sh        (details: bash scripts/apply_verl_patch.sh --check)" >&2
  exit 1
fi
if [ -z "$PYBIN" ] && [ -n "$ENVNAME" ]; then
  PYBIN=$SCRATCH/miniconda3/envs/$ENVNAME/bin/python3
fi
[ -x "${PYBIN:-}" ] || { echo "pass --env NAME (an env under $SCRATCH/miniconda3/envs) or --python PATH." >&2
                         echo "  create one: bash $HERE/setup_env.sh --scratch $SCRATCH --kind train" >&2
                         exit 1; }

if [ -n "$MODEL" ] && [ ! -d "$MODEL" ]; then
  if [ "$DRY" = 1 ]; then
    echo "(--dry-run: $MODEL would be downloaded from Hugging Face into $HF_HOME)"
  else
    echo "resolving Hugging Face repo $MODEL into $HF_HOME"
    RESOLVED=$("$PYBIN" - "$MODEL" <<'PYHF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], ignore_patterns=["*.pth", "*.msgpack", "*.h5", "*.gguf"]))
PYHF
    ) || { echo "could not fetch $MODEL -- is it a repo id, and is huggingface_hub installed in $PYBIN?" >&2
           exit 1; }
    MODEL=$RESOLVED
    echo "  -> $MODEL"
  fi
fi
[ -n "$MODEL" ] || { echo "--model is required: a local directory or a Hugging Face repo id." >&2; exit 1; }
if [ "$DRY" != 1 ] || [ -d "$MODEL" ]; then
  [ -n "$MODEL" ] && [ -f "$MODEL/config.json" ] || {
    echo "no base model: pass --model as a local dir (config.json + tokenizer_config.json) or a Hugging" >&2
    echo "Face repo id such as Qwen/Qwen2.5-3B-Instruct" >&2; exit 1; }
fi

FAMILY=""
case "$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')" in *qwen*) FAMILY=qwen ;; *llama*) FAMILY=llama ;; esac
[ -n "$FAMILY" ] || { echo "cannot tell the model family from $MODEL; pass --name <...qwen...|...llama...>" >&2; exit 1; }
NAME=${NAME:-$FAMILY-$PERT-ckl_${CKL//+/_}-dradd_$DRADD}
case "$NAME" in *qwen*|*llama*) ;; *) echo "--name must contain qwen or llama" >&2; exit 1 ;; esac

: "${PROMPT_LEN:=}"; [ -n "$PROMPT_LEN" ] || { [ "$PERT" = twin ] && PROMPT_LEN=4608 || PROMPT_LEN=2048; }

DATA=${DATA:-$SCRATCH/data/$PERT}
find_pair() {
  for a in train_cp train; do for b in test_cp test; do
    [ -f "$1/$a.parquet" ] && [ -f "$1/$b.parquet" ] && { echo "$1/$a.parquet $1/$b.parquet"; return; }
  done; done
}
PAIR=$(find_pair "$DATA" || true)
if [ -z "$PAIR" ]; then
  if [ -z "$RLLA_JSON" ]; then
    for c in "$SCRATCH"/data/rlla_rl.json "$HERE"/toolrl/dataset/rlla_4k_raw/rlla_rl.json \
             "$SCRATCH"/tmp/rlla_rl.json; do [ -f "$c" ] && { RLLA_JSON=$c; break; }; done
  fi
  [ -f "$RLLA_JSON" ] || { echo "no dataset at $DATA and no --rlla-json to build one from" >&2; exit 1; }
  mkdir -p "$DATA"
  echo "building $PERT dataset from $RLLA_JSON -> $DATA"
  "$PYBIN" "$HERE/canonperm/build_twin_dataset.py" --rlla-json "$RLLA_JSON" --out-dir "$DATA" \
      --tokenizer "$MODEL" --max-prompt-tokens "$PROMPT_LEN"
  PAIR=$(find_pair "$DATA")
  [ -n "$PAIR" ] || { echo "build produced no parquet pair in $DATA" >&2; exit 1; }
fi
TRAIN_FILE=${PAIR%% *}; VAL_FILE=${PAIR##* }

case "$PERT" in twin|order+relabel) VIEW_FAMILY=order+relabel ;; *) VIEW_FAMILY=$PERT ;; esac
LAM_REL=${LAM_REL:-0.02}
LAM_ADD=${LAM_ADD:-0.005}
case "$CKL" in
  rel+add) PERM_LAM=$LAM_REL; DISTR_LAM=$LAM_ADD ;;
esac
if [ "$DRADD" = on ]; then IN_PG=True; else IN_PG=False; fi
case "$DRADD:$CKL" in
  on:*|*:rel+add|*:add) DISTR_ROLLOUT=True ;;
  *)                    DISTR_ROLLOUT=False ;;
esac
CONS_LAM=$(awk -v a="$PERM_LAM" -v b="$DISTR_LAM" 'BEGIN{print (a>b)?a:b}')

: "${TRAIN_BATCH:=512}"; : "${VAL_BATCH:=128}"; : "${PPO_MINI:=128}"; : "${PPO_MICRO:=32}"
: "${ROLLOUT_N:=4}"; : "${SAVE_FREQ:=5}"; : "${TEST_FREQ:=5}"
EPOCHS=${EPOCHS:-15}
: "${MAX_TOKEN_LEN:=12288}"; : "${GRAD_CKPT:=True}"; : "${REF_OFFLOAD:=True}"; : "${ROLLOUT_MEM:=0.45}"
: "${ROLLOUT_TP:=1}"
: "${OPT_OFFLOAD:=False}"
: "${OPT_PARK:=False}"
: "${IDENTITY_SLOTS:=64}"; : "${IDENTITY_PROMPTS:=16}"; : "${N_PERM:=1}"
: "${READOUT_BATCH:=4}"
: "${PREFIX_CACHE:=True}"; : "${GRAD_PROBE_EVERY:=10}"
if [ -z "$GPUS" ]; then
  GPUS=$( { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true; } | wc -l | tr -d ' ')
fi
case "$GPUS" in ''|*[!0-9]*|0) GPUS=1 ;; esac
OUTDIR=${OUTDIR:-$SCRATCH/runs/$NAME}
mkdir -p "$OUTDIR"

export EXPERIMENT_NAME="$NAME"
export WITHLENGTH=0 REFINEDREWARD=0 COARSEREWARD=0 STRICTMATCH=0 CORRECTMAX1=0 MAX1STEP30MAX3=0 \
       SCHEDULEREWARD=0 SCHEDULELENGTH=0 TWIN_PENALTY=${TWIN_PENALTY:-0}
export IDENTITY_VIEW_FAMILY="$VIEW_FAMILY"
export IDENTITY_CLEAN_REDRAW=${CLEAN_REDRAW:-0}
export IDENTITY_LABEL_SPACE=${LABEL_SPACE:-100}
export IDENTITY_DISTRACTOR_REDRAW=${DISTRACTOR_REDRAW:-1}

cat <<EOF
=== $NAME ===
  perturbation    $(printf '%-14s' "$PERT") added-tool interface: $VIEW_FAMILY
  DR-add          $(printf '%-14s' "$DRADD") added-tool rows in the policy gradient: $IN_PG
  CKL             $(printf '%-14s' "$CKL") lambda rel=$PERM_LAM add=$DISTR_LAM
  added-tool pass $DISTR_ROLLOUT
  model           $MODEL
  data            $TRAIN_FILE
                  $VAL_FILE
  run dir         $OUTDIR
  verl            $VERL
  python          $PYBIN
  recipe          batch=$TRAIN_BATCH mini=$PPO_MINI micro=$PPO_MICRO n=$ROLLOUT_N epochs=$EPOCHS
  readout         added-tool $IDENTITY_SLOTS rows/step (whole groups), relabelling $IDENTITY_PROMPTS
                  prompts x1 rollout, $READOUT_BATCH per forward, shared prefix=$PREFIX_CACHE, n_perm=$N_PERM
  memory          gpus=$GPUS max_token_len=$MAX_TOKEN_LEN grad_ckpt=$GRAD_CKPT ref_offload=$REF_OFFLOAD
                  rollout_mem=$ROLLOUT_MEM rollout_tp=$ROLLOUT_TP opt_offload=$OPT_OFFLOAD
                  opt_park=$OPT_PARK
EOF
[ "$DRY" = 1 ] && { echo "(--dry-run: nothing launched)"; exit 0; }

cd "$OUTDIR"
exec "$PYBIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$PROMPT_LEN" \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.actor.max_prompt_length="$PROMPT_LEN" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing="$GRAD_CKPT" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPT_OFFLOAD" \
    +actor_rollout_ref.actor.optimizer_park="$OPT_PARK" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_MEM" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.distractor_rollout="$DISTR_ROLLOUT" \
    +actor_rollout_ref.actor.distractor_in_pg="$IN_PG" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_OFFLOAD" \
    +actor_rollout_ref.actor.consistency_mode=identity \
    +actor_rollout_ref.actor.consistency_coef="$CONS_LAM" \
    +actor_rollout_ref.actor.consistency_identity_perm_coef="$PERM_LAM" \
    +actor_rollout_ref.actor.consistency_identity_distractor_coef="$DISTR_LAM" \
    +actor_rollout_ref.actor.consistency_identity_slots="$IDENTITY_SLOTS" \
    +actor_rollout_ref.actor.consistency_identity_prompts="$IDENTITY_PROMPTS" \
    +actor_rollout_ref.actor.consistency_readout_batch="$READOUT_BATCH" \
    +actor_rollout_ref.actor.consistency_prefix_cache="$PREFIX_CACHE" \
    +actor_rollout_ref.actor.consistency_n_perm="$N_PERM" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=robust-agentic \
    trainer.experiment_name="$NAME" \
    trainer.n_gpus_per_node="$GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$EPOCHS" "$@"
