#!/usr/bin/env bash
set -euo pipefail

NVME=""; PERT=twin; DRADD=off; CKL=rel
MODEL=""; DATA=""; OUTDIR=""; NAME=""; RESUME=""; RLLA_JSON=""; PYBIN=""; ENVNAME=""
EPOCHS=""; GPUS=""; LAM_REL=""; LAM_TWIN=""; DRY=0; CHECK=0
_need() {
  case "${2-}" in
    "") echo "$1 requires a value, and none was given" >&2; exit 2 ;;
    -*) echo "$1 requires a value but got the option '$2' -- an empty shell variable?" >&2; exit 2 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --perturbation) _need "$1" "${2-}"; PERT="$2"; shift 2 ;;
    --dr-add)       _need "$1" "${2-}"; DRADD="$2"; shift 2 ;;
    --ckl)          _need "$1" "${2-}"; CKL="$2"; shift 2 ;;
    --model)        _need "$1" "${2-}"; MODEL="$2"; shift 2 ;;
    --data)         _need "$1" "${2-}"; DATA="$2"; shift 2 ;;
    --rlla-json)    _need "$1" "${2-}"; RLLA_JSON="$2"; shift 2 ;;
    --out)          _need "$1" "${2-}"; OUTDIR="$2"; shift 2 ;;
    --name)         _need "$1" "${2-}"; NAME="$2"; shift 2 ;;
    --resume)       _need "$1" "${2-}"; RESUME="$2"; shift 2 ;;
    --python)       _need "$1" "${2-}"; PYBIN="$2"; shift 2 ;;
    --env)          _need "$1" "${2-}"; ENVNAME="$2"; shift 2 ;;
    --epochs)       _need "$1" "${2-}"; EPOCHS="$2"; shift 2 ;;
    --gpus)         _need "$1" "${2-}"; GPUS="$2"; shift 2 ;;
    --lam-rel)      _need "$1" "${2-}"; LAM_REL="$2"; shift 2 ;;
    --lam-twin)      _need "$1" "${2-}"; LAM_TWIN="$2"; shift 2 ;;
    --scratch)      _need "$1" "${2-}"; [ -z "$NVME" ] || { echo "the scratch directory was given twice: "\
                      "$NVME and $2" >&2; exit 2; }; NVME="$2"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  [ -z "$NVME" ] || { echo "unexpected argument: $1" >&2; exit 2; }; NVME="$1"; shift ;;
  esac
done
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -z "$NVME" ]; then
  echo "usage: bash $0 --scratch SCRATCH_DIR [options]      (also accepted bare, as the first argument)" >&2
  echo "  required, not defaulted: the path differs per machine and a wrong one fills the root volume." >&2
  echo "  It is a property of THIS machine, so it is never guessed. \`df -h\` shows what is mounted here." >&2
  exit 2
fi
[ -d "$NVME" ] && [ -w "$NVME" ] || { echo "FATAL: $NVME is not a writable directory" >&2; exit 1; }
case "$NVME" in /) echo "FATAL: refusing the root volume as scratch" >&2; exit 1 ;; esac
case "$PERT" in twin) ;; *) echo "--perturbation: this arm builds the twin interface only" >&2; exit 2 ;; esac
case "$DRADD" in on|off) ;; *) echo "--dr-add must be on | off" >&2; exit 2 ;; esac
case "$CKL" in rel+twin|twin|rel|none) ;; *) echo "--ckl must be rel+twin | twin | rel | none" >&2; exit 2 ;; esac

export HOME="$NVME/home" TMPDIR="$NVME/tmp" RAY_TMPDIR="$NVME/tmp"
export XDG_CACHE_HOME="$HOME/.cache" PIP_CACHE_DIR="$HOME/.cache/pip" HF_HOME="$HOME/.cache/hf"
export NLTK_DATA="$HOME/.cache/nltk"
export TRITON_CACHE_DIR="$HOME/.cache/triton" VLLM_CACHE_ROOT="$HOME/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/inductor"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME"
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1

if [ -z "$PYBIN" ] && [ -n "$ENVNAME" ]; then
  PYBIN=$NVME/miniconda3/envs/$ENVNAME/bin/python
  [ -x "$PYBIN" ] || { echo "no interpreter at $PYBIN" >&2
                       echo "create the env: bash $HERE/scripts/ckl_verl090_env.sh $NVME" >&2
                       echo "or name one outright: --python /abs/path/to/python" >&2; exit 1; }
fi
[ -x "${PYBIN:-}" ] || { echo "pass --env NAME or --python PATH (see $HERE/scripts/ckl_verl090_env.sh)" >&2; exit 1; }

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export VERL_USE_EXTERNAL_MODULES=ckl.verl_hook
export FLA_CACHE_MODE=${FLA_CACHE_MODE:-full}
export VERL_ENGINE_VENDOR=ckl
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning${PYTHONWARNINGS:+,$PYTHONWARNINGS}"

if [ -n "$MODEL" ] && [ ! -d "$MODEL" ] && [ "$DRY" = 1 ]; then
  echo "(--dry-run: $MODEL would be downloaded into $HF_HOME; not fetching, so the config check needs no model)"
elif [ -n "$MODEL" ] && [ ! -d "$MODEL" ]; then
  echo "resolving Hugging Face repo $MODEL into $HF_HOME"
  MODEL=$("$PYBIN" - "$MODEL" <<'PYHF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], ignore_patterns=["*.pth", "*.msgpack", "*.h5", "*.gguf"]))
PYHF
  ) || { echo "could not fetch the model -- is it a repo id?" >&2; exit 1; }
  echo "  -> $MODEL"
fi
if [ "$DRY" != 1 ] || [ -d "$MODEL" ]; then
  [ -n "$MODEL" ] && [ -f "$MODEL/config.json" ] || {
    echo "pass --model as a local dir (config.json) or a Hugging Face repo id" >&2; exit 1; }
fi

FAMILY=""
case "$(printf '%s%s' "$MODEL" "$NAME" | tr 'A-Z' 'a-z')" in
  *qwen*) FAMILY=qwen ;; *llama*) FAMILY=llama ;;
esac
[ -n "$FAMILY" ] || { echo "cannot tell the model family from '$MODEL'${NAME:+ or --name '$NAME'}." >&2
                     echo "  The reward scorer picks its chat-template split from it and raises on an" >&2
                     echo "  unrecognised one, so pass --name with qwen or llama in it." >&2; exit 1; }
NAME=${NAME:-$FAMILY-$PERT-ckl_${CKL//+/_}-dradd_$DRADD}
case "$NAME" in *qwen*|*llama*) ;; *) echo "--name must contain qwen or llama" >&2; exit 1 ;; esac

: "${PROMPT_LEN:=4608}"
: "${RESPONSE_LEN:=1024}"
: "${MAX_MODEL_LEN:=$((PROMPT_LEN + RESPONSE_LEN))}"

DATA=${DATA:-$NVME/data/$PERT}
find_pair() { for a in train_cp train; do for b in test_cp test; do
    [ -f "$1/$a.parquet" ] && [ -f "$1/$b.parquet" ] && { echo "$1/$a.parquet $1/$b.parquet"; return; }
  done; done; return 0; }
PAIR=$(find_pair "$DATA")
if [ -z "$PAIR" ]; then
  if [ -z "$RLLA_JSON" ]; then
    for c in "$NVME"/data/rlla_rl.json "$HERE"/toolrl/dataset/rlla_4k_raw/rlla_rl.json; do
      [ -f "$c" ] && { RLLA_JSON=$c; break; }; done
  fi
  [ -f "${RLLA_JSON:-}" ] || { echo "no dataset at $DATA and no --rlla-json to build one from" >&2; exit 1; }
  if [ "$DRY" = 1 ]; then
    echo "(--dry-run: the twin dataset would be built from $RLLA_JSON -> $DATA)"
    TRAIN_FILE=$DATA/train.parquet; VAL_FILE=$DATA/test.parquet
    PAIR="$TRAIN_FILE $VAL_FILE"
  else
  mkdir -p "$DATA"
  echo "building the twin dataset from $RLLA_JSON -> $DATA"
  "$PYBIN" "$HERE/canonperm/build_twin_dataset.py" --rlla-json "$RLLA_JSON" --out-dir "$DATA" \
      --tokenizer "$MODEL" --max-prompt-tokens "$PROMPT_LEN"
  PAIR=$(find_pair "$DATA")
  [ -n "$PAIR" ] || { echo "the build produced no parquet pair in $DATA" >&2; exit 1; }
  fi
fi
TRAIN_FILE=${PAIR%% *}; VAL_FILE=${PAIR##* }

LAM_REL=${LAM_REL:-0.02}
LAM_TWIN=${LAM_TWIN:-0.005}
case "$CKL" in
  rel+twin) L_REL=$LAM_REL; L_TWIN=$LAM_TWIN ;;
  twin)     L_REL=0.0;      L_TWIN=$LAM_TWIN ;;
  rel)      L_REL=$LAM_REL; L_TWIN=0.0 ;;
  none)     L_REL=0.0;      L_TWIN=0.0 ;;
esac
if [ "$DRADD" = on ]; then IN_PG=True; else IN_PG=False; fi
case "$DRADD:$CKL" in on:*|*:rel+twin|*:twin) TWIN_IFACE=True ;; *) TWIN_IFACE=False ;; esac

: "${TASKS:=256}"
: "${PPO_MINI:=128}"; : "${ROLLOUT_N:=4}"; : "${SAVE_FREQ:=5}"; : "${TEST_FREQ:=5}"
: "${MAX_TOKEN_LEN:=$((MAX_MODEL_LEN + 512))}"; : "${GRAD_CKPT:=True}"; : "${ROLLOUT_MEM:=0.45}"
if [ "$MAX_TOKEN_LEN" -le "$MAX_MODEL_LEN" ]; then
  echo "FATAL: MAX_TOKEN_LEN=$MAX_TOKEN_LEN must exceed PROMPT_LEN + RESPONSE_LEN = $MAX_MODEL_LEN." >&2
  echo "  A micro-batch cannot hold less than one row, and verl asserts this in the log-prob pass:" >&2
  echo "  \"max_token_len must be greater than the sequence length\". Raise it, or lower PROMPT_LEN." >&2
  exit 1
fi
: "${ENTROPY_COEFF:=0.001}"
: "${ACT_OFFLOAD:=False}"
: "${N_REL:=16}"; : "${N_TWIN:=128}"
if [ "$TWIN_IFACE" = True ]; then TRAIN_BATCH=$((TASKS * 2)); else TRAIN_BATCH=$TASKS; fi
EPOCHS=${EPOCHS:-15}
LR=${LR:-1e-6}
if [ -z "$GPUS" ]; then
  GPUS=$( { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true; } | wc -l | tr -d ' ')
fi
case "$GPUS" in ''|*[!0-9]*|0) GPUS=1 ;; esac
OUTDIR=${OUTDIR:-$NVME/runs/$NAME}
mkdir -p "$OUTDIR"

export EXPERIMENT_NAME="$NAME"
export WITHLENGTH=0 REFINEDREWARD=0 COARSEREWARD=0 STRICTMATCH=0 CORRECTMAX1=0 MAX1STEP30MAX3=0 \
       SCHEDULEREWARD=0 SCHEDULELENGTH=0 TWIN_PENALTY=${TWIN_PENALTY:-0}

VOCAB=0
if [ -f "$MODEL/config.json" ]; then
  VOCAB=$(sed -n 's/.*"vocab_size"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "$MODEL/config.json" | head -1)
  VOCAB=${VOCAB:-0}
fi
if [ "$VOCAB" -gt 0 ]; then
  PEAK="vocab=$VOCAB -> fused head $(awk -v v="$VOCAB" \
      'BEGIN{printf "%.2f", 512*v*4/1073741824}') GiB fp32 per 512-token chunk (constant)"
else
  PEAK="vocab unknown ($MODEL is not downloaded yet, so its config.json cannot be read)"
fi
PEAK="$PEAK
                  one forward <= $MAX_TOKEN_LEN tokens per group (group_tokens_max reports the real one)"

print_banner() {
cat <<EOF
=== $NAME ===
  perturbation    $PERT (twin -> reorder -> relabel, drawn per access)
  DR-add          $DRADD   -> \`I+\` rows in the policy gradient: $IN_PG
  CKL             $CKL     lambda rel=$L_REL add=$L_TWIN
  \`I+\` interface  $TWIN_IFACE
  model           $MODEL
  data            $TRAIN_FILE
                  $VAL_FILE
  run dir         $OUTDIR
  python          $PYBIN
  recipe          tasks/step=$TASKS -> train_batch=$TRAIN_BATCH  mini=$PPO_MINI n=$ROLLOUT_N epochs=$EPOCHS
                  lr=$LR entropy_coeff=$ENTROPY_COEFF reference-KL off
                  prompt=$PROMPT_LEN + response=$RESPONSE_LEN -> vllm max_model_len=$MAX_MODEL_LEN
                  save/test every $SAVE_FREQ
  readout         rollouts read/step: relabelling $N_REL, twin-menu $N_TWIN = $((N_TWIN / (2 * ROLLOUT_N))) tasks x 2 x $ROLLOUT_N
                  group means over $ROLLOUT_N rollouts a side; block width derived
  memory          gpus=$GPUS max_token_len=$MAX_TOKEN_LEN grad_ckpt=$GRAD_CKPT rollout_mem=$ROLLOUT_MEM
                  fla cache mode=$FLA_CACHE_MODE (autotune at training time is OFF)
                  activation offload=$ACT_OFFLOAD
                  $PEAK
EOF
}
if "$PYBIN" -c "import flash_attn" >/dev/null 2>&1; then
  ATTN_IMPL=flash_attention_2
else
  ATTN_IMPL=sdpa
  echo "  flash-attn is not installed in $PYBIN -- attn_implementation=sdpa"
fi

OVERRIDES=(
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL"
    trainer.use_v1=True
    trainer.v1.trainer_mode=ckl_sync
    algorithm.adv_estimator=grpo
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size="$TRAIN_BATCH"
    data.gen_batch_size="$TASKS"
    data.max_prompt_length="$PROMPT_LEN"
    data.max_response_length="$RESPONSE_LEN"
    data.shuffle=True
    data.custom_cls.path=pkg://ckl.dataset
    data.custom_cls.name=CKLTwinDataset
    +data.ckl.twin_interface="$TWIN_IFACE"
    reward.custom_reward_function.path=pkg://ckl.reward
    reward.custom_reward_function.name=compute_score
    actor_rollout_ref.model.path="$MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.enable_gradient_checkpointing="$GRAD_CKPT"
    actor_rollout_ref.model.enable_activation_offload="$ACT_OFFLOAD"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_TOKEN_LEN"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    +ckl.lam_rel="$L_REL"
    +ckl.lam_twin="$L_TWIN"
    +ckl.distractor_in_pg="$IN_PG"
    +ckl.twin_interface="$TWIN_IFACE"
    +ckl.n_rel="$N_REL"
    +ckl.n_twin="$N_TWIN"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.agent.default_agent_loop=ckl_single_turn
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_MEM"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=ckl.verl_hook
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_ENGINE_VENDOR=ckl
    +ray_kwargs.ray_init.runtime_env.env_vars.FLA_CACHE_MODE="$FLA_CACHE_MODE"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONWARNINGS="'$PYTHONWARNINGS'"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$HERE"
    +ray_kwargs.ray_init.runtime_env.env_vars.EXPERIMENT_NAME="$NAME"
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name=robust-agentic
    trainer.experiment_name="$NAME"
    trainer.n_gpus_per_node="$GPUS"
    trainer.nnodes=1
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_epochs="$EPOCHS"
)
print_banner

for _o in "${OVERRIDES[@]}"; do
  _v=${_o#*=}
  case "$_v" in
    *,*)
      case "$_v" in
        \'*\'|\[*\]) ;;
        *) echo "FATAL: override value contains a comma but is not quoted for Hydra:" >&2
           echo "  $_o" >&2
           echo "  Hydra reads an unquoted comma as a list separator. Write key=\"'\$VALUE'\"." >&2
           exit 1 ;;
      esac ;;
  esac
done

for _o in "${OVERRIDES[@]}"; do
  case "$_o" in
    *ckl/*.py)
      echo "FATAL: override names a file inside the ckl package:" >&2
      echo "  $_o" >&2
      echo "  use pkg://ckl.<module> -- a package module can only be loaded as a module." >&2
      exit 1 ;;
  esac
done

[ -n "$RESUME" ] && OVERRIDES+=(trainer.resume_mode=resume_path trainer.resume_from_path="$RESUME")

if [ "$DRY" = 1 ]; then
  echo "(--dry-run: composing the config with Hydra and exiting; nothing is launched)"
  CFG=$TMPDIR/ckl090-config-$$.yaml
  if ! "$PYBIN" -m verl.trainer.main_ppo "${OVERRIDES[@]}" "$@" --cfg job > "$CFG"; then
    echo "the config does NOT compose -- see the Hydra error above" >&2
    exit 1
  fi
  echo "config composes: all ${#OVERRIDES[@]} overrides are real keys  ->  $CFG"
  BASE=$TMPDIR/ckl090-config-stock-$$.yaml
  "$PYBIN" -m verl.trainer.main_ppo --cfg job > "$BASE" 2>/dev/null || {
    echo "could not compose the stock config for comparison" >&2; exit 1; }
  "$PYBIN" - "$CFG" "$BASE" <<'PYCFG' || exit 1
import dataclasses
import sys

from hydra.utils import get_class
from omegaconf import OmegaConf


def flat(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out[prefix + k] = node
            out.update(flat(v, prefix + k + "."))
    return out


mine = flat(OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=False))
base = flat(OmegaConf.to_container(OmegaConf.load(sys.argv[2]), resolve=False))
added = sorted(set(mine) - set(base))
print("  keys this launcher adds: %d" % len(added))

full = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=False)


def target_of(key):
    parts = key.split(".")
    node = full
    best = (None, None)
    for i, part in enumerate(parts[:-1]):
        node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            break
        if "_target_" in node:
            best = (".".join(parts[:i + 1]), node["_target_"])
    return best


bad = 0
for key in added:
    node, target = target_of(key)
    if target is None:
        print("  OK    %-52s (plain config node, no dataclass)" % key)
        continue
    field = key[len(node) + 1:].split(".")[0]
    names = {f.name for f in dataclasses.fields(get_class(target))}
    if field in names:
        print("  OK    %-52s (declared by %s)" % (key, target.rsplit(".", 1)[-1]))
    else:
        bad = 1
        print("  FAIL  %-52s %s declares no `%s`" % (key, target.rsplit(".", 1)[-1], field))
        print("        move it to a node Hydra does not instantiate -- the top-level `ckl.` block")
sys.exit(bad)
PYCFG
  exit 0
fi

cd "$OUTDIR"
exec "$PYBIN" -m verl.trainer.main_ppo "${OVERRIDES[@]}" "$@"
