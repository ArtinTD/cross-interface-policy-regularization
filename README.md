# Cross-Interface Policy Regularization (CIPR)

Post-training and evaluation for **interface generalization**: how much of an agent's task success survives a
semantics-preserving change to the interface it was trained on — renaming the tools, reordering the menu,
adding a plausible near-duplicate. Training adds a consistency-KL term that penalises the policy for behaving
differently across such a change; evaluation measures the surviving accuracy under the same change.

Every entry point is a bash script. There is no bare `python` in this repo's interface: the interpreter lives
in a conda env under the scratch volume, and which env is a property of the machine.

---

# Quickstart

## 0. Set the two variables every command uses

```bash
export CODE=$PWD                  # run from the root of this checkout
export S=$CODE/scratch            # HOME, every cache, env, dataset, checkpoint and output go here
mkdir -p $S
```

`$S` is passed as `--scratch $S` to every script. Nothing is ever written outside it.

## 1. Install

Four envs, one per role. This is all of them; nothing else has to be built.

```bash
git clone https://github.com/qiancheng0/ToolRL.git $CODE/toolrl
git -C $CODE/toolrl checkout 8cee13ec0ca72f0461da372a93a6fd8140dbb840

bash $CODE/setup_env.sh --scratch $S --kind train    # rapt-train: Qwen2.5 training, vendored verl 0.1
bash $CODE/setup_env.sh --scratch $S --kind eval     # rapt-eval:  the BFCL-V4 harness
bash $CODE/setup_env.sh --scratch $S --kind serve    # rapt-serve: the vLLM server eval_open.sh starts
bash $CODE/scripts/ckl_verl090_env.sh --scratch $S   # ckl090:     Qwen3.5 training, verl 0.9.0
```

`--kind` is repeatable if you want several in one call. The Qwen3.5 env is a separate script rather than a
fifth `--kind` because the two verls' pins are mutually exclusive (**The two training arms**, below).
Installing conda is part of the job; on a fresh machine `--conda-only` stops after it, which is how
`ckl_verl090_env.sh` gets the conda it requires to already exist.

## 2. Train

One entry point per model family, and both of them are here. Qwen2.5 is the prose/ReAct surface, Qwen3.5 is
native function calling; the surface follows the family in training and evaluation alike.

```bash
# Qwen2.5 arm — vendored verl 0.1 + verl_patch/ overlay
nohup bash $CODE/train.sh --scratch $S --env rapt-train --model Qwen/Qwen2.5-7B-Instruct \
      > $S/tmp/train.log 2>&1 &
echo $!

# Qwen3.5 arm — verl 0.9.0
nohup bash $CODE/train_ckl090.sh --scratch $S --env ckl090 --model Qwen/Qwen3.5-4B \
      > $S/tmp/train-ckl090.log 2>&1 &
echo $!
```

Defaults are the reported configuration: twin interface, DR-add on (`off` on the 0.9.0 arm), CKL `rel+add`
(`rel` there), 105 steps (15 epochs at batch 512 / mini 128 / n 4). The dataset builds on first use. Nothing
about a long run may be foreground — a dropped ssh session kills it. Each run uses every visible GPU, so
start the second when the first has finished.

## 3. Evaluate

```bash
# Qwen2.5: the step-100 checkpoint of the train.sh run above
CKPT=$S/runs/qwen-twin-ckl_rel_add-dradd_on/checkpoints/robust-agentic/qwen-twin-ckl_rel_add-dradd_on/actor/global_step_100
nohup bash $CODE/eval_open.sh --scratch $S --model $CKPT --tag qwen-twin-s100 \
      > $S/tmp/eval-bfcl.log 2>&1 &
```

`eval_open.sh` owns the whole chain and skips any cell already done, so it is re-runnable. It prints the
table at the end. It serves the model on every visible GPU, so run one evaluation at a time.

When that run has finished, add the in-distribution cell (twins and reordering) to the data tree it built,
and evaluate it with canonical names:

```bash
$S/miniconda3/envs/rapt-eval/bin/python $CODE/perturb/build_bfcl_twin.py \
      --clean $S/bfcl_data/all/clean/normal --out $S/bfcl_data/all
nohup bash $CODE/eval_open.sh --scratch $S --model $CKPT --tag qwen-twin-s100 --types twin --variants canon \
      > $S/tmp/eval-bfcl-twin.log 2>&1 &
```

Qwen3.5 checkpoints are FSDP shards. Merge one into a Hugging Face directory, then evaluate it with native
function calling:

```bash
bash $CODE/scripts/ckl_merge_ckpt.sh --scratch $S --env ckl090 \
     --ckpt $S/runs/qwen-twin-ckl_rel-dradd_off/checkpoints/robust-agentic/qwen-twin-ckl_rel-dradd_off/global_step_100
nohup bash $CODE/eval_open.sh --scratch $S --model $S/models/merged-qwen-twin-ckl_rel-dradd_off-global_step_100 \
      --tag qwen35-twin-s100 --fc > $S/tmp/eval-bfcl-35.log 2>&1 &
```

---

# Training commands

## `setup_env.sh` — build a pinned conda env

```
bash $CODE/setup_env.sh --scratch DIR --kind train|eval|serve [--kind ...]
```

Assumes nothing is installed: if conda is absent it is installed too, into `$S/miniconda3`. Re-running is
safe — an existing env is reused and only missing packages are installed.

| flag | value | default | what it does |
|---|---|---|---|
| `--scratch` | dir | **required** | the data volume. HOME and every cache are repointed into it |
| `--kind` | `train` \| `eval` \| `serve` | **required** (or `--conda-only`) | which env to build; repeatable |
| `--conda-only` | — | off | install miniconda into `$S` and stop. What a fresh machine needs before `scripts/ckl_verl090_env.sh` |
| `--conda` | dir | `$S/miniconda3` | an existing conda install to use instead |
| `--bfcl-repo` | dir | cloned | an existing BFCL checkout for `--kind eval` |
| `--recreate` | — | off | delete and rebuild the env rather than topping it up |

| `--kind` | env built | python | for |
|---|---|---|---|
| `train` | `rapt-train` | 3.10 | `train.sh` — GRPO on the vendored verl 0.1 |
| `eval` | `rapt-eval` | 3.10 | the BFCL-V4 harness (`bfcl generate/evaluate`); no CUDA needed |
| `serve` | `rapt-serve` | 3.11 | the vLLM server `eval_open.sh` starts |

`ckl090` is not built here — see `scripts/ckl_verl090_env.sh` below.

Its env var:

| env var | default | effect |
|---|---|---|
| `BFCL_REPO` | `$S/bench/gorilla` | where the BFCL clone lands, for `--kind eval`. `--bfcl-repo` is the same thing as a flag |

## `train.sh` — the Qwen2.5 arm (vendored verl 0.1 + `verl_patch/` overlay)

```
bash $CODE/train.sh --scratch DIR --env NAME --model MODEL [options]
```

| flag | value | default | what it does |
|---|---|---|---|
| `--scratch` | dir | **required** | the data volume |
| `--model` | HF repo id or local dir | **required** | resolved to a local snapshot, so tokenizer, family check, vLLM and any later resume all see one path |
| `--env` | conda env name | **required** (or `--python`) | the env under `$S/miniconda3/envs/` |
| `--python` | path to a python | — | an interpreter named outright, instead of `--env` |
| `--lam-rel` | float | `0.02` | the relabelling term's λ |
| `--lam-add` | float | `0.005` | the added-tool term's λ |
| `--name` | string | `<family>-<perturbation>-ckl_<terms>-dradd_<on\|off>` | run name; **must carry the model family** (`qwen`/`llama`) — the reward function picks its chat-template split from it |
| `--data` | dir | `$S/data/<perturbation>/` | a prebuilt dataset directory instead of building one |
| `--rlla-json` | file | auto-discovered (incl. `$S/tmp/rlla_rl.json`) | the ToolRL source JSON the dataset is built from |
| `--out` | dir | `$S/runs/<NAME>/` | the run directory (the trainer's cwd) |
| `--epochs` | int | `15` | 15 with the rest of the recipe **is** 105 steps; changing it makes the run incomparable |
| `--gpus` | int | every visible card | how many ranks to launch |
| `--verl` | dir | `$CODE/toolrl` | the vendored verl checkout |
| `--dry-run` | — | off | compose the config, check every override is a real Hydra key, run the unit suite; download nothing, launch nothing |

If the `verl_patch/` overlay is missing, `train.sh` refuses to start rather than quietly training stock GRPO:
every consistency flag is a key unpatched verl accepts and never reads. Apply it with
`bash $CODE/scripts/apply_verl_patch.sh` (`--check` reports drift).

### The recipe — do not change these

ToolRL's public recipe, set in the script and reproduced by omitting every flag. Every one is an env-var
override, and every override makes the run incomparable with the rest of the table.

| env var | value |
|---|---|
| `TRAIN_BATCH` | `512` |
| `VAL_BATCH` | `128` |
| `PPO_MINI` | `128` |
| `PPO_MICRO` | `32` |
| `ROLLOUT_N` | `4` |
| `EPOCHS` | `15` → 105 steps |
| `SAVE_FREQ` | `5` |
| `TEST_FREQ` | `5` |

Three more are in the script body and not env vars at all: lr `1e-6`, `entropy_coeff=0.001`, response length
`1024`.

### Knobs that change the measurement — sign-off, not tuning

Every env var below decides *what* is estimated, so moving one makes a different quantity. Complete:

| env var | default | what it sets |
|---|---|---|
| `IDENTITY_SLOTS` | `64` | added-tool rows the readout reads per step, in whole groups |
| `IDENTITY_PROMPTS` | `16` | prompts the relabelling term reads per step, one rollout each |
| `N_PERM` | `1` | relabellings drawn per prompt |
| `LABEL_SPACE` | `100` | how many canonical labels the dataset draws a row's names from |

The other reward switches are pinned off in the script and are not knobs: `WITHLENGTH`, `REFINEDREWARD`,
`COARSEREWARD`, `STRICTMATCH`, `CORRECTMAX1`, `MAX1STEP30MAX3`, `SCHEDULEREWARD`, `SCHEDULELENGTH`.

### Capacity knobs — env vars that change cost, not results

Set them in front of the command. Each changes how the same update is computed, not what it computes.

| env var | default | effect |
|---|---|---|
| `MAX_TOKEN_LEN` | `12288` | tokens per micro-batch; grad-accum above the `PROMPT_LEN + RESPONSE_LEN` floor |
| `READOUT_BATCH` | `4` | slots sharing one readout forward, so passes per mini-batch = (`IDENTITY_PROMPTS` + `IDENTITY_SLOTS`) / this. Per-pass cost is ~92% fixed FSDP traversal, so this sets the readout's wall time. Under `PREFIX_CACHE=True` the peak does not scale with it; under `False` the peak *is* one batch's graph and this is what to halve |
| `ROLLOUT_TP` | `1` | vLLM tensor parallelism. Raise it when the weights approach the rollout budget (a 14B is 28 GiB) |
| `ROLLOUT_MEM` | `0.45` | vLLM `gpu_memory_utilization`. Below the resident weights every sequence is preempted to CPU swap |
| `OPT_OFFLOAD` | `False` | Adam moments to host between stages (verl's own flag). 13.75 GiB at 14B, a tensor move and not a host-side optimizer — but loaded across the whole policy update, so it does **not** cover the backward's peak |
| `GRAD_CKPT` | `True` | gradient checkpointing (**required** by the shared-prefix readout, non-reentrant) |
| `REF_OFFLOAD` | `True` | reference-policy params to host |
| `PROMPT_LEN` | `4608` under `twin`, else `2048` | prompt cap |

### Backend env vars — already at a working value

All three, each set to what one measured failure required. Change one at a time and prove a step.

| env var | default | why |
|---|---|---|
| `NCCL_NET_PLUGIN` | `none` | the EFA plugin segfaults rank 0 on a single node |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | what lets the readout's large allocations succeed on a heap the rollout has already fragmented |
| `VLLM_ATTENTION_BACKEND` | `XFORMERS` | this vLLM build's own pin. Switching it alongside CUDA-graph capture produced an illegal memory access inside generation on 8×H100 |

`ALLOW_BASE_MISMATCH=1` is read by `scripts/apply_verl_patch.sh`, not by `train.sh`: it lets the overlay
apply onto a vendored verl whose base file differs from the one the patch was cut against.

## `train_ckl090.sh` — the Qwen3.5 arm (verl 0.9.0, no patch)

```
bash $CODE/train_ckl090.sh --scratch DIR --env ckl090 --model Qwen/Qwen3.5-4B [options]
```

Its own flags, in full — the overlap with `train.sh` is large but not total, and the differences are not
guessable from that table.

| flag | value | default | what it does |
|---|---|---|---|
| `--scratch` | dir | **required** | the data volume. Also accepted as the first positional argument |
| `--model` | HF repo id or local dir | **required** | resolved to a local snapshot |
| `--env` | conda env name | **required** (or `--python`) | the env under `$S/miniconda3/envs/` |
| `--python` | path to a python | — | an interpreter named outright, instead of `--env` |
| `--perturbation` | `twin` | `twin` | this arm builds the twin interface only and refuses any other value |
| `--dr-add` | `on` \| `off` | `off` | whether the added-tool rows also enter the policy gradient |
| `--ckl` | `rel+twin` \| `rel` \| `twin` \| `none` | `rel` | which consistency terms are on. The added-tool term is called `twin` here |
| `--lam-rel` | float | `0.02` | the relabelling term's λ |
| `--lam-twin` | float | `0.005` | the added-tool term's λ |
| `--name` | string | `<family>-twin-ckl_<terms>-dradd_<on\|off>` | run name; **must carry the model family** |
| `--data` | dir | `$S/data/twin/` | a prebuilt dataset directory instead of building one |
| `--rlla-json` | file | auto-discovered | the ToolRL source JSON the dataset is built from |
| `--out` | dir | `$S/runs/<NAME>/` | the run directory |
| `--resume` | dir | — | a `global_step_N` checkpoint directory to continue from |
| `--epochs` | int | `15` | 15 with the rest of the recipe **is** 105 steps |
| `--gpus` | int | every visible card | how many ranks to launch |
| `--dry-run` | — | off | compose the config and check every override; launch nothing |

There is no `--verl`: the env's installed verl 0.9.0 is the one used, and `ckl/` reaches it through
registered hooks rather than a file overlay.

Its recipe, fixed the same way `train.sh`'s is:

| env var | value |
|---|---|
| `TASKS` | `256` source tasks per step. The batch is `TASKS × 2` under the twin interface, because each task is submitted as two prompts |
| `PPO_MINI` | `128` |
| `ROLLOUT_N` | `4` |
| `EPOCHS` | `15` → 105 steps |
| `SAVE_FREQ` | `5` |
| `TEST_FREQ` | `5` |
| `ENTROPY_COEFF` | `0.001` |

Its measurement knobs and capacity knobs, complete:

| env var | default | effect |
|---|---|---|
| `N_REL` | `16` | rollouts the relabelling term reads per step, each read twice (its own naming, and its relabelled rewrite) |
| `N_TWIN` | `128` | rollouts the added-tool term reads per step. A rollout count, not a task count: the task count is derived as `N_TWIN / (2 × ROLLOUT_N)`, so it does not silently double when `ROLLOUT_N` does |
| `PROMPT_LEN` | `4608` | prompt cap |
| `RESPONSE_LEN` | `1024` | response cap |
| `MAX_MODEL_LEN` | `PROMPT_LEN + RESPONSE_LEN` | the served context |
| `MAX_TOKEN_LEN` | `MAX_MODEL_LEN + 512` | tokens per micro-batch of the actor update; pure gradient accumulation |
| `GRAD_CKPT` | `True` | gradient checkpointing |
| `ROLLOUT_MEM` | `0.45` | vLLM `gpu_memory_utilization` |
| `ACT_OFFLOAD` | `False` | **must stay off.** verl's activation offloading keeps per-layer-group bookkeeping for one forward, and the readout issues a second inside the same step: `KeyError: 55` from inside the model. It buys ~3 GiB when it works |
| `FLA_CACHE_MODE` | `full` | fla's Triton kernels look up the shipped configs instead of autotuning at training time. The default `disabled` benchmarks candidates while the training graph is resident — an OOM inside `autotuner.benchmark` |
| `NCCL_NET_PLUGIN` | `none` | as on the other arm |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | as on the other arm |

## `scripts/ckl_merge_ckpt.sh` — FSDP shards → a servable HF directory

```
bash $CODE/scripts/ckl_merge_ckpt.sh --scratch DIR --env ckl090 --ckpt .../global_step_N [--out DIR]
```

**Only the Qwen3.5 arm needs this.** verl 0.9.0's `save_contents` is `model, optimizer, extra`, so
`global_step_N/actor/` holds `model_world_size_8_rank_*.pt` plus a `huggingface/` dir with the config and
tokenizer only — no weights, and vLLM pointed at it fails on a missing safetensors index. The merged
directory is the script's last line; pass it to `eval_open.sh --model`. The checkpoint is untouched,
re-running is free, and reassembly is verl's own merger. `train.sh` saves a full HF directory at every
checkpoint, so its checkpoints serve as-is.

---

# Evaluation commands

## `eval_open.sh` — BFCL-V4 matrix for an open-weight checkpoint

```
bash $CODE/eval_open.sh --scratch DIR --model CKPT [--tag TAG] [options]
```

Owns the whole chain, skipping any part already done: a private clone of the BFCL install → the two harness
patches → the model registered in PROMPT mode with the RLLA handler → the perturbation data, built and gated
→ a vLLM server on this machine → the cells → the table.

| flag | value | default | what it does |
|---|---|---|---|
| `--scratch` | dir | **required** | the data volume |
| `--model` | dir or repo id | **required** | the checkpoint to serve. Ignored by `--score-only` |
| `--tag` | string, **no underscore** | `basename(--model)` with `_`→`-` | names the install, the output tree and the registry entry. `bfcl evaluate` un-escapes `_` to `/` and dies after generation has been paid for |
| `--types` | quoted list, or `ALL` | `ALL` | which perturbations to run — values below |
| `--variants` | quoted list | `normal canon canon_sw` | which name conditions — values below |
| `--parts` | comma list, or `all` | all 10 (n=2501) | the gold-bearing BFCL partitions, all of them: `simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`, `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`. Name a subset only deliberately, and then spell it out in the command that produced the number |
| `--decoy-instruction` | `default` \| `budget` \| `rbtc` \| `inventory` \| `inventory+budget` | `default` | the sentence the six reward types append to the query — values below |
| `--server` | url | starts its own vLLM | reuse an existing endpoint; starts and stops nothing |
| `--gpu` | `0,1,...` | every visible card | pin the replicas to a subset |
| `--port` | int | auto | the vLLM port |
| `--threads` | int | 32 per replica | harness client threads |
| `--score-only` | — | off | re-score a finished run: no model, no GPU |
| `--out` | dir | `$S/bfcl_out/<TAG>` | results tree |
| `--data` | dir | `$S/bfcl_data/<parts>` | a prebuilt perturbation tree |
| `--src-data` | dir | `$S/bfcl_pristine` | the pristine BFCL data the tree is built from |
| `--bfcl-repo` / `--bfcl-bin` / `--serve-bin` | dir / path / path | discovered under `$S` | override a located component |
| `--fc` | — | off | serve with the tool-call parser and register the model in native function-calling mode; for Qwen3.5 checkpoints |
| `--tool-parser` | name | `qwen3_coder` | the vLLM tool-call parser used with `--fc` |

GPUs are detected: every visible card becomes a vLLM replica (`--data-parallel-size N`) behind one base URL,
and a cell is thousands of independent requests, so replicas are what buys throughput. Torn down on exit,
including Ctrl-C. `--max-model-len` is never passed to the server — a window below the model's native context
truncates long prompts into 400s and deflates every score into something non-comparable.

### `--types`: the 13 perturbations, plus `clean`

Space-separated and quoted (`--types "clean cost_decoy"`), or `ALL`. `clean` is the unperturbed baseline
cell, not a perturbation. The MDP category says which part of the interface each one touches.

| `--types` value | display name | category | what it changes |
|---|---|---|---|
| `clean` | Clean | — | nothing; the baseline every drop is measured against |
| `query_typos` | Typo | Observation | realistic typos in the user query; tools and gold unchanged |
| `redundant` | RedunTool | Action | one useless-but-similar tool injected: a different tool's description and parameters under a gold-sibling name; gold unchanged |
| `same_name_A` | Dup-NoDesc | Action | a same-name twin of each needed tool, blank description, empty params |
| `same_name_B` | Dup-Desc | Action | same twin, same description, empty params |
| `same_name_C` | Dup-WrongP | Action | blank description, another tool's params |
| `same_name_D` | Dup-DescWP | Action | same description, another tool's params |
| `same_name_E` | Dup-SwapDP | Action | another tool's description **and** params |
| `cost_decoy` | Cost decoy | Reward | the correct tool is **renamed** and priced low; a clone keeping the original name is priced high. `[Cost: $X/call]`, appended |
| `cost_decoy_nt` | Cost decoy (nat. lang.) | Reward | same, as `(costs approximately $X per call)` |
| `cost_decoy_abbrev` | Cost decoy (abbrev) | Reward | same, abbreviated tool name + `[Cost: $X/call]`, prepended |
| `latency_decoy` | Latency decoy | Reward | same construction on latency: `[Response time: ~Xms]` |
| `latency_decoy_nt` | Latency decoy (nat. lang.) | Reward | `(typically responds in Xms)` |
| `latency_decoy_abbrev` | Latency decoy (abbrev) | Reward | abbreviated name + `[Latency: Xms]`, prepended |
| `twin` | Twins & reordering | — | the training perturbation: a degraded twin of every tool, and the tools reordered. Built by `perturb/build_bfcl_twin.py` into an existing data tree; run with `--variants canon` |

In the five `same_name_*` types the gold is **unchanged** — the name clash is the perturbation. In the six
reward types the gold **relabels**: the renamed tool is the correct answer, so the familiar-named clone is
the decoy. Report these by display name, never by the raw code.

### `--variants`: the name conditions

| value | what it is |
|---|---|
| `normal` | the perturbation's own names, as it produced them |
| `canon` | canonical conversion on top of the perturbation: every function name replaced by `function_N` |
| `canon_sw` | `canon` with the decoy declared first. Built for the six reward types only, and it has no `clean` cell |

### `--decoy-instruction`: what the reward types append to the query

| value | the sentence |
|---|---|
| `default` | the explicit objective ("…spending as few dollars as possible: whenever two tools would both do the job, call the cheaper one"), with `_abbrev` keeping RBTC's short phrasings. **Every reported number** |
| `budget` | that explicit objective for all six, `_abbrev` included |
| `rbtc` | RBTC's own short phrasings for all six ("I have a limited budget for this task.") |
| `inventory` | "Before making a decision, give a short record of every tool, what it does, and its important properties." — **and nothing about preferring one** |
| `inventory+budget` | the inventory request, then the objective |

`inventory` states no objective, so the query no longer determines which of the two schema-identical tools is
correct: that cell measures whether the model prefers the cheaper tool *unprompted*, which is a different
quantity and is not comparable with the other columns. The validity gate reports it as `objective_unstated`
— expected under `inventory`, a defect under anything else.

The wording is part of the data, so it is part of both directory names (`$S/bfcl_data/<parts>-inventory`,
`$S/bfcl_out/<TAG>-inventory`), and two further checks close the gaps naming alone leaves:

- the built tree records its wording in `decoy_instruction.txt`; a run whose `--decoy-instruction` disagrees
  with the tree it was pointed at **stops** instead of measuring the other wording;
- each cell records the md5 of the data its responses were generated against (`result/.data`). A cell whose
  data has changed is **refused before the skip**, rather than re-scored against new gold.

### Two properties of every number

- **The AST checker is fixed before anything is scored**, in one pass (`scripts/bfcl_fix_checker.sh`). It
  repairs two defects that made the score a fact about declaration order: single-tool categories validated
  against the *first declared* tool instead of the reference answer, and a duplicated name resolving to
  whichever declaration came first. Both fixes are unconditional and both are no-ops on an unperturbed
  interface — `clean` and `normal`/`canon` cells are byte-identical before and after. Each cell records its
  fix version in `score/.checker`, and the report tooling refuses a cell that predates it.
- A tag must contain no underscore (see `--tag`).

---

# Outputs

## Paths

All under `--scratch`, written `$S`. Nothing lands outside it.

| what | path |
|---|---|
| run directory (the trainer's cwd) | `$S/runs/<NAME>/` |
| checkpoints, Qwen2.5 | `$S/runs/<NAME>/checkpoints/robust-agentic/<NAME>/actor/global_step_N/` |
| checkpoints, Qwen3.5 | `$S/runs/<NAME>/checkpoints/robust-agentic/<NAME>/global_step_N/actor/` |
| merged checkpoint (servable) | `$S/models/merged-<NAME>-global_step_N/`, or `--out` |
| built training set | `$S/data/<perturbation>/{train,test}.parquet` — one per family, reused |
| model snapshot | `$S/home/.cache/hf/hub/models--<org>--<model>/snapshots/<sha>/` |
| BFCL results | `$S/bfcl_out/<TAG>/<variant>/<type>/` |
| BFCL harness clone (per tag) | `$S/bfcl_installs/<TAG>/berkeley-function-call-leaderboard/` |
| BFCL perturbed data | `$S/bfcl_data/<parts>/` |
| BFCL pristine source | `$S/bfcl_pristine/` — the verified upstream tree both are built from |
| vLLM server log | `$S/tmp/serve_<TAG>.log` |
| conda + envs | `$S/miniconda3/`, `envs/{rapt-train,rapt-eval,rapt-serve,ckl090}` |
| upstream checkouts | `$S/bench/gorilla/` (BFCL), `$S/bench/verl-090/` |
| `HOME` and every cache | `$S/home/`, `$S/home/.cache/{pip,hf,triton,vllm,inductor}` |

No script redirects its own output, so a training or eval log is wherever you sent it.

## A BFCL cell: three directories

| directory | contents |
|---|---|
| `result/` | the generations: id + response, plus `.data` (the md5 of the data they were generated against) |
| `score/` | the aggregate, the wrong records only, and `.checker` (the checker-fix version) |
| `review/<TAG>/BFCL_v4_<part>.jsonl` | one record per item, ready to read |

A `review/` record:

```json
{"reward": 0, "id": "multiple_2", "query": "…",
 "prompt": {"messages": [...every message, every role...],
            "tools":    [...every declared tool, full schema...],
            "injected": [1]},
 "response": "…", "error": [...], "gold": [...]}
```

`reward` is 1/0 — `score/` lists only the wrong records, so an id absent from it was right. `error` and
`gold` appear on a wrong record only. `prompt` is the whole prompt verbatim: `tools` is the record's own
`function` list with nothing reshaped — full descriptions (where the reward families' `[Cost: $X/call]`
annotation lands), parameter properties, types and `required` — and `messages` every message at every role.
A summarised menu is not a substitute: the perturbations act on the schemas, so a field left out is a change
that cannot be seen. `injected` is the index list of the tools the perturbation added.

`review/` exists because **the query is in neither `result/` nor `score/`**, and a cell is run by swapping
data files *into* the harness install, so afterwards the install holds whatever the last cell swapped in and
nothing on disk says what an earlier cell used.

---

# Envs

One env per role, each built from a pinned spec in `env/`. Every version in those files is `==`: a range
means two builds of the same env name are two different runtimes. Build them on the machine, into `$S`.

| env | build it with | pinned by | for |
|---|---|---|---|
| `rapt-train` | `setup_env.sh --kind train` | `env/rapt-train.txt` + `toolrl/requirements.txt` | **Qwen2.5 training** — `train.sh` |
| `ckl090` | `scripts/ckl_verl090_env.sh` | `env/ckl090.txt` | **Qwen3.5 training** — `train_ckl090.sh`, and `scripts/ckl_merge_ckpt.sh` |
| `rapt-eval` | `setup_env.sh --kind eval` | `env/bfcl.txt` | the BFCL-V4 harness (no CUDA) |
| `rapt-serve` | `setup_env.sh --kind serve` | `env/serve.txt` | the vLLM server `eval_open.sh` starts |

The two training envs are mutually exclusive and must stay separate. The Qwen3.5 build installs
`env/ckl090.txt` before verl, so verl's own ranges are already satisfied and resolve to those versions;
flash-attn, fla-core, flash-linear-attention and causal-conv1d are pinned on their install lines in the
script instead, because they are compiled against that torch and have to go after it.

# The two training arms

They cannot share a process: verl 0.9.0 needs `vllm>=0.18` and `transformers>=5.5`; the vendored verl 0.1
needs `vllm<=0.6.3` and `transformers<4.48`. They share the algorithm and the reward
scorer, nothing else.

| | `train.sh` | `train_ckl090.sh` |
|---|---|---|
| verl | vendored 0.1 + `verl_patch/` overlay | 0.9.0, no patch — eight seams, four registrations |
| model family | Qwen2.5 | Qwen3.5 |
| surface | ToolRL prose menu (ReAct-style prompt) | Qwen3.5 native function calling (XML) |
| env | `rapt-train` | `ckl090` |
| trainer | `verl_patch/verl/workers/actor/dp_actor.py` | `ckl/` |
| checkpoints | full HF dir, serve as-is | FSDP shards, need `ckl_merge_ckpt.sh` |

The surface follows the model family, in training **and** evaluation. They are not interchangeable.

# Every script in `scripts/`

The entry points above call these; each is also runnable on its own. All of them:

| script | what it does |
|---|---|
| `apply_verl_patch.sh` | copy `verl_patch/` over the vendored verl in `toolrl/`, idempotently. `--check` reports drift; `train.sh` refuses to start on drift |
| `bfcl_build_data.sh` | build every BFCL ladder data cell: clean + the 13 types, each in `normal` and `canon`, plus `canon_sw` for the six reward types |
| `bfcl_seed_pristine.sh` | seed a verified pristine BFCL single-turn tree from upstream, data and gold together |
| `bfcl_fix_checker.sh` | the two AST-checker fixes, unconditional and idempotent, applied before anything is scored |
| `bfcl_register_model.py` | register one model with a BFCL install, in PROMPT mode, idempotently |
| `bfcl_sweep.sh` | run the matrix cells for one already-served model: swap the data and gold file in together, generate, score |
| `ckl_verl090_env.sh` | build the `ckl090` env |
| `ckl090_fast_kernels.sh` | install the linear-attention fast path into an existing `ckl090` env, and prove it took |
| `ckl_merge_ckpt.sh` | FSDP shards → a servable HF directory. Qwen3.5 arm only |

# Repo layout

Every top-level entry, and what it is for. The first group is the current experiment.

```
train.sh                   Qwen2.5 training entry point (vendored verl 0.1 + verl_patch/)
train_ckl090.sh            Qwen3.5 training entry point (verl 0.9.0 + ckl/)
eval_open.sh               BFCL-V4 interface matrix for an open-weight checkpoint
setup_env.sh               conda + one pinned env per role
scripts/                   what the entry points call; the table above lists all of it
ckl/                       the consistency method against verl 0.9.0's hooks
verl_patch/                the consistency trainer as a file-for-file overlay on the vendored verl
toolrl/                    ToolRL/verl 0.1, cloned at the pinned commit (Install)
canonperm/                 the training-set builder, the BFCL canonicalizer and the two BFCL model handlers
perturb/                   the 13 BFCL perturbation types (rbtc16.py) with their validity gate, and the
                           in-distribution twin cell (build_bfcl_twin.py)
report/                    the BFCL result table and the per-item review
env/                       the pinned specs the env builders install from
README.md                  this file
```
