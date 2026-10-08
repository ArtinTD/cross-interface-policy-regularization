# Cross-Interface Policy Regularization (CIPR)

Code for training tool-calling agents with CIPR and evaluating them on BFCL v4 under held-out interface
perturbations.

| | Qwen2.5 | Qwen3.5 |
|---|---|---|
| trainer | ToolRL's verl 0.1 with the files in `verl_patch/` copied over it | verl 0.9.0 with the extension package `ckl/` |
| entry point | `train.sh` | `train_ckl090.sh` |
| tool-call format | ToolRL prompt format | native function calling |

Both build the same training set (`canonperm/build_twin_dataset.py`): ToolRL's 4,000 examples with canonical
names (`function_NN`, `arg_NN`) and, for every example, perturbed interfaces built by twin expansion, tool and
argument reordering, and relabeling.

Names in the code and their counterparts in the paper:

| code | paper |
|---|---|
| relabelling term (`identity_perm`; `rel` in `--ckl`) | rescoring KL |
| added-tool term (`identity_distractor`; `twin` in `--ckl`) | rollout KL |
| DR-add (`distractor_in_pg`, `--dr-add on`) | multi-interface training |
| twin, distractor | degraded twin of a tool |
| `ckl` | consistency KL |

`train.sh` always trains with both terms and multi-interface training; `train_ckl090.sh` selects them with
`--ckl` and `--dr-add`.

## Setup

All environments, caches and outputs go under one directory, `$S`.

```bash
S=/path/to/scratch
git clone https://github.com/qiancheng0/ToolRL.git toolrl
git -C toolrl checkout 8cee13ec0ca72f0461da372a93a6fd8140dbb840
bash setup_env.sh --scratch $S --kind train --kind eval --kind serve   # Qwen2.5 training, BFCL, vLLM
bash scripts/ckl_verl090_env.sh --scratch $S                          # Qwen3.5 training
```

`setup_env.sh` installs Miniconda under `$S` if needed, copies `verl_patch/` over `toolrl/`, and clones BFCL at the
commit used for all results. Package versions are pinned in `env/`. The runs used 8 A100 40GB GPUs.

## Training

The training set is built on the first run from `toolrl/dataset/rlla_4k_raw/rlla_rl.json`.

```bash
# Qwen2.5-3B
bash train.sh --scratch $S --env rapt-train --model Qwen/Qwen2.5-3B-Instruct

# Qwen3.5-4B
bash train_ckl090.sh --scratch $S --env ckl090 --model Qwen/Qwen3.5-4B --ckl rel+twin --dr-add on
```

Checkpoints are written every 5 updates under `$S/runs/<name>/checkpoints/robust-agentic/<name>/`: as Hugging Face
directories in `actor/global_step_N` for Qwen2.5, and as FSDP shards in `global_step_N/actor` for Qwen3.5.
The settings in the scripts are sample configurations; the coefficients are `--lam-rel` and `--lam-add`
(`--lam-twin` for Qwen3.5), and every other setting is an environment variable read at the top of each script.

Qwen3.5 checkpoints are FSDP shards; convert one to a Hugging Face directory before evaluation:

```bash
bash scripts/ckl_merge_ckpt.sh --scratch $S --env ckl090 --ckpt <.../global_step_N>
```

## Evaluation

`eval_open.sh` serves a checkpoint with vLLM on all visible GPUs, builds the perturbed BFCL data on first use, runs
every cell and prints the table.

```bash
bash eval_open.sh --scratch $S --model <checkpoint> --tag qwen25-cipr          # Qwen2.5, prompt format
bash eval_open.sh --scratch $S --model <merged checkpoint> --tag qwen35-cipr --fc   # Qwen3.5, function calling
```

Add `--decoy-instruction inventory` for the implicit version of the cost and latency perturbations. Results are in
`$S/bfcl_out/<tag>/<condition>/<type>/`, where the condition is `normal` (original names), `canon` (canonical
names) or `canon_sw` (canonical names with the costlier or slower equivalent listed first).

The in-distribution evaluation applies the training perturbation (twins and reordering) to every BFCL item. Build
its cell into the data tree created by the first evaluation, then evaluate it with canonical names:

```bash
python3 perturb/build_bfcl_twin.py --clean $S/bfcl_data/all/clean/normal --out $S/bfcl_data/all
bash eval_open.sh --scratch $S --model <checkpoint> --tag qwen25-cipr --types twin --variants canon
```
