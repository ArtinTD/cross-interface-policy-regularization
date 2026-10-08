import copy
import logging
import math
import uuid

import torch
from omegaconf import open_dict
from tensordict import TensorDict

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

from verl.trainer.ppo.padding_utils import (build_padding_position_ids, build_padding_routed_experts,
                                            upsample_batch_to_divisible_size)
from verl.trainer.ppo.v1 import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from . import rowbuild
from .loss import CKLConfig


def _verl_partition(workloads, k):
    return get_seqlen_balanced_partitions(
        calculate_workload(torch.tensor(workloads, dtype=torch.int64)), k_partitions=k, equal_size=True)


def _cfg(config):
    node = config.get("ckl", None) or {}
    return {"lam_rel": float(node.get("lam_rel", 0.0)),
            "lam_twin": float(node.get("lam_twin", 0.0)),
            "distractor_in_pg": bool(node.get("distractor_in_pg", True)),
            "twin_interface": bool(node.get("twin_interface", True)),
            "n_rel": int(node.get("n_rel", 16)),
            "n_twin": int(node.get("n_twin", 128)),
            "seed": int(node.get("seed", 0))}


def _pad_row(values, width=rowbuild.MAX_MENU, fill=-1):
    return torch.tensor(list(values) + [fill] * (width - len(values)), dtype=torch.long)


def _draw(rng, items, n):
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    keep = set(rng.sample(range(len(items)), n))
    return [x for j, x in enumerate(items) if j in keep]


@register_trainer("ckl_sync")
class CKLTrainer(PPOTrainerSync):

    def __init__(self, config, *args, **kwargs):
        with open_dict(config):
            config.trainer.v1.trainer_mode = "sync"
        super().__init__(config, *args, **kwargs)
        self.ckl = _cfg(self.config)
        self._read_seq = 0
        self._unit_seq = 0
        self._tok_cache = None

    def init(self, *args, **kwargs):
        out = super().init(*args, **kwargs)
        self._install_loss_fn()
        self._check_alignment()
        return out


    def _install_loss_fn(self):
        from functools import partial

        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config.actor import ActorConfig

        from .loss import ckl_loss

        if max(self.ckl["lam_rel"], self.ckl["lam_twin"]) <= 0.0 and self.ckl["distractor_in_pg"]:
            return
        actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)

        cfg = CKLConfig(lam_rel=self.ckl["lam_rel"], lam_twin=self.ckl["lam_twin"],
                        distractor_in_pg=self.ckl["distractor_in_pg"],
                        group_size=int(self.config.actor_rollout_ref.rollout.n))
        self.actor_rollout_wg.set_loss_fn(partial(ckl_loss, config=actor_config, ckl=cfg))

    def _check_alignment(self):
        fg = self.config.algorithm.get("filter_groups", None)
        if fg is not None and fg.get("enable", False):
            logging.getLogger(__name__).warning(
                "algorithm.filter_groups.enable is set: DAPO filtering discards groups, and a "
                "discarded `I` leaves its `I+` with nothing to compare against. The CKL term "
                "will lose those pairs -- watch ckl_pairs.")
        if self.config.trainer.v1.sampler.get("sync_refill_failed_groups", False):
            logging.getLogger(__name__).warning(
                "trainer.v1.sampler.sync_refill_failed_groups is set: it forces gen_batch_size "
                "to 1 and refills by prompt count, which cannot preserve the `I`/`I+` pairing. "
                "Refilled groups will not contribute to the CKL term -- watch ckl_pairs.")
        if self.ckl["twin_interface"] and self.config.data.train_batch_size != 2 * self._gen_batch_size():
            gen = self._gen_batch_size()
            why = ""
            if gen == 1:
                why = ("\n  data.gen_batch_size has been REWRITTEN to 1 by verl's exact-refill path. It fires "
                       "when\n  trainer.v1.trainer_mode != 'sync', algorithm.filter_groups.enable=True, or "
                       "trainer.v1.sampler.sync_refill_failed_groups=True.\n  This arm needs whole-batch "
                       "fetches: each task is submitted as two prompts and both must land in one batch.")
            raise ValueError("with the twin interface on, data.train_batch_size must be exactly twice "
                             "data.gen_batch_size (%d vs %d): each task is submitted as two prompts, `I` and "
                             "`I+`, and sync mode trains on exactly what it submitted.%s"
                             % (self.config.data.train_batch_size, gen, why))
        if not self._reads_anything():
            return
        w = self._group_width()
        actor = self.config.actor_rollout_ref.actor
        if actor.get("shuffle", False):
            raise ValueError("actor_rollout_ref.actor.shuffle must be False: the mini-batch iterator would "
                             "permute rows and split a reading's block across two optimizer steps")
        if actor.get("ulysses_sequence_parallel_size", 1) != 1:
            raise ValueError("the readout gathers at absolute packed positions; run with "
                             "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1")
        fused_backend = (self.config.actor_rollout_ref.model.get("fused_kernel_options", None)
                         or {}).get("impl_backend", "torch")
        if fused_backend != "torch":
            raise ValueError("model.fused_kernel_options.impl_backend must be 'torch', not %r: the readout "
                             "takes the hidden state and the output head off FusedLinearForPPO, which only "
                             "the torch backend's forward calls" % fused_backend)
        if not self.config.actor_rollout_ref.model.get("use_fused_kernels", False):
            raise ValueError(
                "model.use_fused_kernels must be True. Without it verl's forward materialises a full "
                "(tokens x vocab) logits tensor, and the readout's gather PINS it for the whole backward -- "
                "37 GiB for one micro-batch of readout rows. With it the forward calls FusedLinearForPPO, "
                "which is where ckl/verl_hook.py takes the hidden state and the output head so the readout "
                "can apply that head at its three read positions.")
        mini = actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        dp = self._actor_dp_size()
        if mini % (dp * w):
            raise ValueError("ppo_mini_batch_size * rollout.n = %d must be a multiple of dp_size * "
                             "group_width = %d * %d, or a block straddles two mini-batches"
                             % (mini, dp, w))

    def _group_width(self):
        per_rank = (self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    * self.config.actor_rollout_ref.rollout.n) // self._actor_dp_size()
        need = 2 * int(self.config.actor_rollout_ref.rollout.n)
        for w in range(need, per_rank + 1):
            if per_rank % w == 0:
                return w
        raise ValueError("no group width divides the per-rank mini-batch of %d rows while covering the %d "
                         "rows a reading can need; raise ppo_mini_batch_size or rollout.n" % (per_rank, need))

    def _reads_anything(self):
        return max(self.ckl["lam_rel"], self.ckl["lam_twin"]) > 0.0

    def _gen_batch_size(self):
        return self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size

    def _actor_dp_size(self):
        wg = self.actor_rollout_wg
        if "actor" not in wg._dispatch_info:
            wg._dispatch_info["actor"] = wg._query_dispatch_info("actor")
        return max(wg._dispatch_info["actor"]) + 1

    def _tokens(self):
        if self._tok_cache is None:
            from .tokens import build
            self._tok_cache = build(self.tokenizer)
        return self._tok_cache


    def _next_train_batch(self, num_prompts=None):
        want = num_prompts if num_prompts is not None else self.config.data.train_batch_size
        tasks = max(1, (want + 1) // 2)
        batch = super()._next_train_batch(tasks)
        return self._duplicate_onto_twin_interface(batch)

    def _add_prompts_to_generate(self, num_prompts: int) -> int:
        tasks = max(1, (int(num_prompts) + 1) // 2)
        return self._submit_batch_to_rollout(self._next_train_batch(2 * tasks))

    @staticmethod
    def _duplicate_onto_twin_interface(batch: TensorDict) -> TensorDict:
        n = len(batch)
        drop = {"ckl_twin_ok", "ckl_twin_prompt", "ckl_twin_tools", "ckl_twin_gold", "ckl_twin_numbers",
                "ckl_twin_n_tools"}
        rows_out = []
        for half in (0, 1):
            for i in range(n):
                src = batch[i]
                row = {k: src[k] for k in batch.keys() if k not in drop}
                if half:
                    ok = int(batch["ckl_twin_ok"][i])
                    if ok:
                        row["raw_prompt"] = batch["ckl_twin_prompt"][i]
                        row["ckl_tools"] = batch["ckl_twin_tools"][i]
                        row["ckl_numbers"] = batch["ckl_twin_numbers"][i]
                        row["ckl_n_tools"] = batch["ckl_twin_n_tools"][i]
                        rm = row.get("reward_model")
                        row["reward_model"] = dict(rm or {}, ground_truth=batch["ckl_twin_gold"][i])
                    row["ckl_interface"] = torch.ones((), dtype=torch.long)
                    row["uid"] = str(uuid.uuid4())
                rows_out.append(row)
        out = list_of_dict_to_tensordict(rows_out)
        tu.assign_non_tensor_data(out, "global_steps", int(batch["global_steps"]))
        return out


    def _balance_batch(self, batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        if not self._reads_anything():
            return super()._balance_batch(batch, metrics, logging_prefix, keep_minibatch)
        n_pg = len(batch)
        batch = self._append_readings(batch, metrics)
        n_read = len(batch) - n_pg
        w = self._group_width()
        dp = self._actor_dp_size()
        multiple = math.lcm(self._get_required_batch_multiple(dp), w * dp)
        batch = upsample_batch_to_divisible_size(batch, multiple, self.tokenizer.eos_token_id)
        seq_lens = [int(t["seq_len"]) for t in batch.tags]
        reading_blocks = ()
        if w and n_read and n_pg % w == 0 and n_read % w == 0:
            reading_blocks = range(n_pg // w, (n_pg + n_read) // w)
        metrics["ckl/reading_blocks_placed"] = float(len(reading_blocks) if reading_blocks else 0)
        order = list(rowbuild.block_order(seq_lens, w, dp, _verl_partition, reading_blocks))
        batch.reorder(order)
        metrics["%s/blocks" % logging_prefix] = len(seq_lens) // w
        blocks = [sum(seq_lens[b * w:(b + 1) * w]) for b in range(len(seq_lens) // w)]
        metrics["%s/group_tokens_max" % logging_prefix] = max(blocks)
        metrics["%s/group_tokens_mean" % logging_prefix] = sum(blocks) / len(blocks)
        return batch


    def _append_readings(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        fields = ["input_ids", "prompts", "ckl_numbers", "ckl_n_tools", "ckl_interface", "ckl_pair_id",
                  "ckl_readable"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        table, span = self._tokens()
        w = self._group_width()
        pad = [bool(t.get("is_padding", False)) for t in batch.tags]
        readable = data["ckl_readable"].reshape(-1).tolist()
        iface = data["ckl_interface"].reshape(-1).tolist()

        rng = self._reading_rng()
        cand_rel = [i for i in range(len(batch)) if not pad[i] and readable[i] and iface[i] == 0]
        want_rel = _draw(rng, cand_rel, self.ckl["n_rel"]) if self.ckl["lam_rel"] > 0 else []
        G = int(self.config.actor_rollout_ref.rollout.n)
        units, skips = [], {}
        want_twin = []
        if self.ckl["lam_twin"] > 0:
            by_pair = {}
            for i in range(len(batch)):
                if pad[i] or not readable[i]:
                    continue
                by_pair.setdefault(int(data["ckl_pair_id"][i]), {}).setdefault(int(iface[i]), []).append(i)
            both = sorted(p for p, sides in by_pair.items() if sides.get(0) and sides.get(1))
            metrics["ckl/twin_groups_available"] = len(both)
            if not both:
                skips["no_task_on_both_interfaces"] = skips.get("no_task_on_both_interfaces", 0) + 1
            n_tasks = max(1, self.ckl["n_twin"] // (2 * G))
            metrics["ckl/twin_groups_wanted"] = n_tasks
            want_twin = _draw(rng, both, n_tasks)

        n_rel_read = 0
        for i in want_rel:
            ids, plen, nums = self._row_span(data, i)
            if nums is None:
                skips["no_naming"] = skips.get("no_naming", 0) + 1
                continue
            func, arg = rowbuild.draw_perms(rng, table.space)
            rowset, why = rowbuild.build(ids, plen, nums, table, span, func, arg, width=w,
                                         read_id=self._next_read_id())
            if rowset is None:
                skips[why] = skips.get(why, 0) + 1
                continue
            units.append(self._tag(rowset, self._next_unit_id(), -1))
            n_rel_read += 1

        n_groups, members = 0, 0
        for pair in want_twin:
            uid, rowsets, kept = self._next_unit_id(), [], [0, 0]
            for side in (0, 1):
                for i in by_pair[pair][side][:G]:
                    ids, plen, nums = self._row_span(data, i)
                    if nums is None:
                        skips["no_naming"] = skips.get("no_naming", 0) + 1
                        continue
                    rowset, why = rowbuild.build_plain(ids, plen, nums, table, span, width=w,
                                                       read_id=self._next_read_id())
                    if rowset is None:
                        skips[why] = skips.get(why, 0) + 1
                        continue
                    rowsets += self._tag(rowset, uid, side)
                    kept[side] += 1
            if not (kept[0] and kept[1]):
                skips["group_lost_a_side"] = skips.get("group_lost_a_side", 0) + 1
                continue
            units.append(rowsets)
            n_groups += 1
            members += kept[0] + kept[1]

        for k, v in skips.items():
            metrics["ckl/skip_%s" % k] = metrics.get("ckl/skip_%s" % k, 0) + v
        metrics["ckl/units"] = len(units)
        metrics["ckl/rel_readings"] = n_rel_read
        metrics["ckl/twin_groups"] = n_groups
        metrics["ckl/twin_members_per_side"] = (members / (2.0 * n_groups)) if n_groups else 0.0
        metrics["ckl/rollouts_read"] = float(n_rel_read + members)
        if not units:
            return batch
        blocks, too_wide = rowbuild.pack_blocks(units, w)
        if too_wide:
            metrics["ckl/skip_unit_wider_than_%d" % w] = len(too_wide)
        rows_total = sum(len(b) for b in blocks)
        metrics["ckl/rows"] = rows_total
        metrics["ckl/rows_padding"] = rows_total - sum(len(u) for u in units)
        return self._write_rows(batch, blocks)

    @staticmethod
    def _tag(rowset, unit, side):
        for r in rowset:
            r["unit"], r["side"] = unit, side
        return rowset

    def _reading_rng(self):
        import random
        return random.Random(self.ckl["seed"] * 1000003 + self.global_steps)

    def _next_read_id(self):
        self._read_seq += 1
        return self._read_seq

    def _next_unit_id(self):
        self._unit_seq += 1
        return self._unit_seq

    @staticmethod
    def _row_span(data, i):
        ids = data["input_ids"][i].tolist()
        plen = int(data["prompts"][i].shape[0])
        nums = [int(x) for x in data["ckl_numbers"][i].tolist() if int(x) >= 0]
        n = int(data["ckl_n_tools"][i])
        return ids, plen, (nums[:n] if n and len(nums) >= n else None)

    def _write_rows(self, batch: KVBatchMeta, blocks) -> KVBatchMeta:
        template = tq.kv_batch_get(keys=[batch.keys[0]], partition_id=batch.partition_id)[0]
        keys, tags, fields = [], [], []
        uid = "ckl%s" % uuid.uuid4().hex
        for b, block in enumerate(blocks):
            for r, row in enumerate(block):
                unit, side = row.get("unit", -1), row.get("side", -1)
                sample = {}
                for k in template.keys():
                    v = template[k]
                    sample[k] = v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                ids = torch.tensor(row["ids"], dtype=torch.int64)
                plen = max(1, min(int(row["plen"]), ids.numel() - 1))
                attn = torch.ones_like(ids, dtype=torch.int64)
                zero = torch.zeros(ids.numel() - plen, dtype=torch.int64)
                sample.update(prompts=ids[:plen], responses=ids[plen:], input_ids=ids,
                              attention_mask=attn,
                              position_ids=build_padding_position_ids(template.get("position_ids"), attn),
                              response_mask=zero, loss_mask=zero.clone(),
                              rm_scores=zero.to(torch.float32),
                              rollout_log_probs=zero.to(torch.float32), num_turns=0,
                              uid="%s_%d" % (uid, b),
                              ckl_read_id=torch.tensor(row["read_id"], dtype=torch.long),
                              ckl_role=torch.tensor(row["role"], dtype=torch.long),
                              ckl_cont_digit=torch.tensor(row["cont_digit"], dtype=torch.long),
                              ckl_gate_pos=torch.tensor(row["gate_pos"], dtype=torch.long),
                              ckl_tens_pos=_pad_row(row["tens_pos"]),
                              ckl_numbers=_pad_row(row["numbers"]),
                              ckl_unit=torch.tensor(unit if row["read_id"] >= 0 else -1,
                                                    dtype=torch.long),
                              ckl_side=torch.tensor(side if row["read_id"] >= 0 else -1,
                                                    dtype=torch.long),
                              ckl_synthetic=torch.tensor(1, dtype=torch.long))
                if "multi_modal_inputs" in sample:
                    sample["multi_modal_inputs"] = {}
                re_ = build_padding_routed_experts(template.get("routed_experts"), ids.numel())
                if re_ is not None:
                    sample["routed_experts"] = re_
                else:
                    sample.pop("routed_experts", None)
                keys.append("%s_%d_%d" % (uid, b, r))
                tags.append(dict(copy.deepcopy(batch.tags[0]), is_padding=True, prompt_len=plen,
                                 response_len=ids.numel() - plen, seq_len=ids.numel()))
                fields.append(sample)

        tq.kv_batch_put(keys=keys, partition_id=batch.partition_id,
                        fields=list_of_dict_to_tensordict(fields), tags=tags)
        n = len(batch)
        zeros = torch.zeros(n, dtype=torch.long)
        neg = torch.full((n,), -1, dtype=torch.long)
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id,
                        fields=TensorDict({"ckl_read_id": neg.clone(), "ckl_role": neg.clone(),
                                           "ckl_cont_digit": neg.clone(), "ckl_gate_pos": zeros.clone(),
                                           "ckl_tens_pos": neg.unsqueeze(1).repeat(1, rowbuild.MAX_MENU),
                                           "ckl_numbers": neg.unsqueeze(1).repeat(1, rowbuild.MAX_MENU),
                                           "ckl_unit": neg.clone(), "ckl_side": neg.clone(),
                                           "ckl_synthetic": zeros.clone()}, batch_size=n))
        return KVBatchMeta(keys=batch.keys + keys, tags=batch.tags + tags,
                           partition_id=batch.partition_id, fields=batch.fields,
                           extra_info=batch.extra_info)


