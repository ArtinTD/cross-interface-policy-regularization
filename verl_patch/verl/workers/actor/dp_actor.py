# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import functools
import itertools
import json
import os
import random
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis


def _sibling(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_IDK = _sibling('identity_kl')
_LIX = _sibling('label_index')
_GP = _sibling('grad_probe')

_LABEL_SPACE = 100
_LOG_KL_CLAMP = 10.0
_SEED_STRIDE = 1000003



class _LabelTokens:

    def __init__(self, tokenizer, pool=None, fc=False):
        self.pool = list(pool) if pool else None
        self.width = 3 if self.pool else 4
        digits = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
        if digits != list(range(digits[0], digits[0] + 10)):
            raise RuntimeError("the ten digits are not consecutive token ids: %s" % digits)
        self.digit_lo, self.digit_hi = digits[0], digits[0] + 9
        self.und = tokenizer.convert_tokens_to_ids('_')
        if tokenizer.convert_ids_to_tokens(self.und) != '_':
            raise RuntimeError("'_' is not a single token in this tokenizer")
        xml_f, xml_a = ('<function=%s_07>', '<parameter=%s_07>') if fc else (None, None)
        self.func = self._prefixes(tokenizer, 'function', xml_f)
        self.arg = self._prefixes(tokenizer, 'arg', xml_a)
        self.label_ids = None
        self.space = _LABEL_SPACE
        self.label_of = {}


    def _prefixes(self, tokenizer, name, xml_tpl=None):
        def site(ids):
            return [i for i in range(1, len(ids) - 2)
                    if ids[i] == self.und and ids[i + 1] == self.digit_lo
                    and ids[i + 2] == self.digit_lo + 7]

        out = []
        for tpl in ('"%s_07"', ' %s_07'):
            ids = tokenizer(tpl % name, add_special_tokens=False)['input_ids']
            hits = site(ids)
            if len(hits) != 1:
                raise RuntimeError("%r does not tokenize as one [prefix, '_', tens, units] site: %s"
                                   % (tpl % name, ids))
            out.append(ids[hits[0] - 1])
        return tuple(dict.fromkeys(out))


def _label_site_grid(table, ids):
    _B, L = ids.shape
    f, u = ids[:, 0:L - 3], ids[:, 1:L - 2]
    a, b = ids[:, 2:L - 1], ids[:, 3:L]
    body = ((u == table.und) & (a >= table.digit_lo) & (a <= table.digit_hi)
            & (b >= table.digit_lo) & (b <= table.digit_hi))
    label = (a - table.digit_lo) * 10 + (b - table.digit_lo)

    def prefixed(toks):
        m = f == toks[0]
        for t in toks[1:]:
            m = m | (f == t)
        return m & body

    return prefixed(table.func), prefixed(table.arg), label.clamp(min=0)


def _relabel_by_lut(table, ids, func_lut, arg_lut=None):
    out = ids.clone()
    _B, L = ids.shape
    if L < table.width:
        return out
    func_site, arg_site, label = _label_site_grid(table, ids)
    idx = label.clamp(0, table.space - 1)
    tens, units = out[:, 2:L - 1], out[:, 3:L]
    for site, lut in ((func_site, func_lut), (arg_site, arg_lut)):
        if lut is None:
            continue
        new = lut[idx]
        tens.copy_(torch.where(site, table.digit_lo + torch.div(new, 10, rounding_mode='floor'), tens))
        units.copy_(torch.where(site, table.digit_lo + (new % 10), units))
    return out


@functools.lru_cache(maxsize=1)
def _cache_classes():
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache

    class _Capture(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)
                self.value_cache.append(None)
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            return key_states, value_states

    class _Frozen(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            k, v = self.key_cache[layer_idx], self.value_cache[layer_idx]
            return torch.cat([k, key_states], dim=-2), torch.cat([v, value_states], dim=-2)

    return _Capture, _Frozen


class _Seqs:

    def __init__(self, ids, mask, p_end):
        if ids.shape != mask.shape or not 0 < p_end < ids.size(-1):
            raise RuntimeError("readout sequences need matching ids and mask and a prompt region inside the "
                               "row: %s vs %s, p_end %d" % (tuple(ids.shape), tuple(mask.shape), p_end))
        self._src = ids
        self._ids = ids.cpu()
        self._plen = mask[:, :p_end].sum(-1).tolist()
        self._rlen = mask[:, p_end:].sum(-1).tolist()
        self._p_end = p_end

    def __getitem__(self, i):
        pl, rl = self._plen[i], self._rlen[i]
        return self._ids[i, self._p_end - pl:self._p_end + rl].tolist(), pl

    def row(self, i):
        pl, rl = self._plen[i], self._rlen[i]
        return self._src[i, self._p_end - pl:self._p_end + rl]


def _expand_cache(captured, take, frozen_cls):
    if not getattr(captured, 'key_cache', None) or any(k is None for k in captured.key_cache):
        raise RuntimeError("the prefix forward left its key/value cache empty, so the tails would attend "
                           "nothing; this transformers version does not fill a handed-in cache")
    out = frozen_cls()
    out.key_cache = [k.index_select(0, take) for k in captured.key_cache]
    out.value_cache = [v.index_select(0, take) for v in captured.value_cache]
    return out


_STRUCTURAL_TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>",
                    "<function=", "</function>", "<parameter=", "</parameter>",
                    "<tools>", "</tools>", "<tool_response>", "</tool_response>")


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        tokenizer=None,
    ):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer
        self._max_prompt_len = int(self.config.get('max_prompt_length', 0) or 0)
        self._opt_park = bool(self.config.get('optimizer_park', False))
        self._cons = self._consistency_config()
        self._cons_metrics = {}
        self._labels = None
        self._grad_probe = None
        self._cons_step = 0
        self._cons_passes = None
        self._nfwd = 0
        self._prof = {}
        self._adv_writer = None
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)


    def _consistency_config(self):
        get = self.config.get
        coef = float(get('consistency_coef', 0.0))
        micro_bsz = int(get('consistency_micro_bsz', 4))
        kl_hi = float(get('consistency_kl_logratio_hi', 2.0))
        c = SimpleNamespace(
            coef=coef,
            mode=get('consistency_mode', 'render'),
            present_only=get('consistency_v5', False),
            micro_bsz=micro_bsz,
            n_perm=int(get('consistency_n_perm', 3)),
            n_dup=int(get('consistency_n_dup', 3)),
            n_order=int(get('consistency_n_order', 3)),
            perm_args=get('consistency_perm_args', False),
            label_weight=float(get('consistency_label_weight', 1.0)),
            only=get('consistency_only', False),
            perm_coef=float(get('consistency_perm_coef', coef)),
            dup_coef=float(get('consistency_dup_coef', coef)),
            order_coef=float(get('consistency_order_coef', coef)),
            decision_coef=float(get('consistency_decision_coef', coef)),
            decision_slots=int(get('consistency_decision_micro_bsz', micro_bsz)),
            decision_kl_cap=float(get('consistency_decision_kl_cap', kl_hi)),
            identity_perm_coef=float(get('consistency_identity_perm_coef', coef)),
            identity_distractor_coef=float(get('consistency_identity_distractor_coef', coef)),
            identity_slots=int(get('consistency_identity_slots', micro_bsz)),
            identity_prompts=int(get('consistency_identity_prompts', 0) or 0),
            harvest=bool(get('consistency_harvest', False)),
            kl_lo=float(get('consistency_kl_logratio_lo', -10.0)),
            kl_hi=kl_hi,
            label_pool=int(get('consistency_label_pool', 0)),
            label_alphabet=get('consistency_label_alphabet', ''),
            perm_attempts=int(get('consistency_perm_attempts', 16)),
            max_tools=int(get('consistency_max_tools', 16)),
            pad_row_len=int(get('consistency_pad_row_len', 4)),
            readout_batch=int(get('consistency_readout_batch', 8)),
            prefix_cache=bool(get('consistency_prefix_cache', False)),
            kl_deadband=float(get('consistency_kl_deadband', 0.0)),
            kl_upper=float(get('consistency_kl_upper', 0.0)),
            kl_upper_coef=float(get('consistency_kl_upper_coef', 10.0)),
            upper_warm=int(get('consistency_upper_warm', 0)),
            upper_rate=float(get('consistency_upper_rate', 0.0)),
            upper_coef_lo=float(get('consistency_upper_coef_lo', 1.0)),
            upper_coef_hi=float(get('consistency_upper_coef_hi', 50.0)),
            upper_fmt_floor=float(get('consistency_upper_fmt_floor', 0.85)),
            upper_kl_tol=float(get('consistency_upper_kl_tol', 0.02)),
            lam_stall_rate=float(get('consistency_lam_stall_rate', 0.0)),
            lam_stall_warm=int(get('consistency_lam_stall_warm', 0)),
            lam_stall_fmt_floor=float(get('consistency_lam_stall_fmt_floor', 0.85)),
            lam_stall_kl_tol=float(get('consistency_lam_stall_kl_tol', 0.02)),
            separate_update=bool(get('consistency_separate_update', False)),
            separate_lr=float(get('consistency_separate_lr', 0.0)),
            distractor_share_target=float(get('consistency_distractor_share_target', 0.0)),
            share_rate=float(get('consistency_share_rate', 1.25)),
            share_format_floor=float(get('consistency_share_format_floor', 0.85)),
            share_lam_lo=float(get('consistency_share_lam_lo', 0.0002)),
            share_lam_hi=float(get('consistency_share_lam_hi', 0.02)),
            grad_probe_every=int(get('grad_probe_every', 0)),
            profile=bool(get('consistency_profile', False)),
        )
        if c.label_pool and c.label_pool < 2:
            raise ValueError("consistency_label_pool must be 0 (the whole alphabet) or at least 2")
        c.any_coef = max(coef, c.perm_coef, c.dup_coef, c.order_coef, c.decision_coef,
                         c.identity_perm_coef, c.identity_distractor_coef)
        return c

    def _label_tokens(self):
        if self._labels is None:
            if self.tokenizer is None:
                raise RuntimeError("the consistency terms need the actor's tokenizer")
            pool = None
            path = getattr(self._cons, 'label_alphabet', '')
            if path:
                with open(path) as f:
                    pool = json.load(f)
                if not isinstance(pool, list) or not all(isinstance(x, str) for x in pool):
                    raise ValueError("%s must hold a JSON list of label strings" % path)
            table = _LabelTokens(self.tokenizer, pool, fc=self._fc_native())
            self._verify_relabelling(table)
            self._labels = table
        return self._labels

    def _verify_relabelling(self, table):
        f_from, f_to, a_from, a_to = '01', '03', '02', '04'
        text = ('{"name": "function_%s", "arguments": {"arg_%s": 1}} function_%s arg_%s'
                % (f_from, a_from, f_from, a_from))
        ids = torch.tensor([self.tokenizer(text, add_special_tokens=False)['input_ids']],
                           dtype=torch.long)
        func_lut, arg_lut = (torch.arange(table.space, dtype=torch.long),
                             torch.arange(table.space, dtype=torch.long))
        func_lut[1], arg_lut[2] = 3, 4
        got = self.tokenizer.decode(_relabel_by_lut(table, ids, func_lut, arg_lut)[0].tolist())
        want = (text.replace('function_%s' % f_from, 'function_%s' % f_to)
                    .replace('arg_%s' % a_from, 'arg_%s' % a_to))
        if got != want:
            raise RuntimeError("relabelling a rendered label did not rename it: %r became %r, wanted %r"
                               % (text, got, want))
        return


    def prof_metrics(self, prefix=''):
        out = {'actor/prof/' + prefix + k: v for k, v in self._prof.items()}
        self._prof = {}
        return out

    def _forward_micro_batch(self, micro_batch, temperature, need_entropy=True, harvest=None):
        self._nfwd += 1
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)
                logits_rmpad = output.logits.squeeze(0)

                if not self.use_ulysses_sp:
                    pos_in_row = indices % seqlen
                    keep = ((pos_in_row >= seqlen - response_length - 1)
                            & (pos_in_row <= seqlen - 2))
                    logits_rmpad = logits_rmpad[keep]
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled[keep]
                    indices = indices[keep]

                logits_rmpad.div_(temperature)


                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad) if need_entropy else None

                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if need_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen) if need_entropy else None
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
                entropy = (full_entropy.squeeze(-1)[:, -response_length - 1:-1] if need_entropy
                           else torch.zeros_like(log_probs))

            else:
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = (verl_F.entropy_from_logits(logits) if need_entropy
                           else torch.zeros_like(log_probs))

            return entropy, log_probs


    def _decode_text(self, ids):
        keep = getattr(self, '_keep_special_ids', None)
        if keep is None:
            keep = set()
            for t in _STRUCTURAL_TAGS:
                enc = self.tokenizer(t, add_special_tokens=False)['input_ids']
                if len(enc) == 1:
                    keep.add(int(enc[0]))
            self._keep_special_ids = keep
            self._drop_special_ids = {int(i) for i in (self.tokenizer.all_special_ids or [])
                                      if int(i) not in keep}
        if not keep:
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        return self.tokenizer.decode([i for i in ids if int(i) not in self._drop_special_ids],
                                     skip_special_tokens=False)


    @staticmethod
    def _fc_needs_tools(meta, tools):
        return isinstance(meta, dict) and meta.get('tools') and not tools


    def _as_meta(self, meta):
        if isinstance(meta, str):
            meta = json.loads(meta)
        return meta if isinstance(meta, dict) else None


    def _fc_native(self):
        if getattr(self, '_fc_native_t', None) is None:
            tools = [{"type": "function", "function": {
                "name": "function_01", "description": "d",
                "parameters": {"type": "object", "properties": {}, "required": []}}}]
            try:
                text = self.tokenizer.apply_chat_template([{"role": "user", "content": "u"}], tools=tools,
                                                          tokenize=False, add_generation_prompt=True)
            except Exception:
                text = ""
            self._fc_native_t = "<function=" in text
        return self._fc_native_t

    def _gate_token(self):
        if getattr(self, '_gate_id', None) is None:
            self._gate_id = self._single_token('<tool_call>')
        return self._gate_id

    def _close_token(self):
        if getattr(self, '_close_id', None) is None:
            self._close_id = self._single_token('</tool_call>')
        return self._close_id

    def _think_ids(self):
        if getattr(self, '_think_t', None) is None:
            ids = self.tokenizer('</think>', add_special_tokens=False)['input_ids']
            self._think_t = tuple(ids or ())
        return self._think_t

    def _single_token(self, text):
        tid = self.tokenizer.convert_tokens_to_ids(text)
        if tid is None or self.tokenizer.convert_ids_to_tokens(tid) != text:
            raise RuntimeError("%s is not a single token in this tokenizer" % text)
        return tid

    def _probe_ids(self):
        if getattr(self, '_probe_t', None) is None:
            opening = ('\n\n<tool_call>\n<function=function_' if self._fc_native()
                       else '\n<tool_call>\n{"name": "function_')
            ids = self.tokenizer(opening, add_special_tokens=False)['input_ids']
            if self._gate_token() not in ids:
                raise RuntimeError("the readout's probe %r does not tokenize to contain <tool_call>: %r"
                                   % (opening, ids))
            self._probe_t = ids
        return self._probe_t

    def _read_cols(self, dev):
        if getattr(self, '_read_cols_t', None) is None:
            table = self._label_tokens()
            labels = torch.arange(table.digit_lo, table.digit_hi + 1, dtype=torch.long)
            self._read_cols_t = torch.cat(
                [labels, torch.tensor([self._gate_token()], dtype=torch.long)]).to(dev)
            self._digit_cols = torch.arange(labels.numel(), dtype=torch.long, device=dev)
            self._gate_token_col = int(labels.numel())
        return self._read_cols_t

    def _render_rollout(self, resp_text, sys, user, tools=None):
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        prefix = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                    add_generation_prompt=True)
        plen = len(self.tokenizer(prefix, add_special_tokens=False)['input_ids'])
        ids = self.tokenizer(prefix + resp_text, add_special_tokens=False)['input_ids']
        cap = self._max_prompt_len
        if cap and plen > cap:
            ids, plen = ids[plen - cap:], cap
        return self._span(ids, plen)

    def _span(self, ids, plen):
        return _LIX.rollout_span(ids, plen, self._gate_token(), self._close_token(), self._think_ids(),
                                 self._probe_ids(), self._label_tokens())

    def _read_batch(self, specs, temperature, dev, want_resp=True):
        pad_id = self.tokenizer.pad_token_id or 0
        table = self._label_tokens()
        digit_lo = table.digit_lo
        single = table.width == 3
        for s in specs:
            assert s is None or s[2], "the readout needs at least one label site"
        rows, queries, resps, at = _LIX.readout_batch(
            [None if s is None else (len(s[0]), s[1], s[2], 0 if single else len(s[3]),
                                     -1 if single else _LIX.own_tens(s[0], s[2], s[3], digit_lo))
             for s in specs], want_resp=want_resp, single=single)
        pad_len = max(int(self._cons.pad_row_len), 2)
        width = max([rl for _s, _t, rl in rows] + [pad_len])
        built = []
        for s, ti, rl in rows:
            ids, _gate, tens_positions, needed = specs[s]
            v = list(ids) if ti < 0 else _LIX.digit_row(ids, tens_positions,
                                                        sorted(needed)[ti], digit_lo)
            v = v[:rl]
            built.append(v + [pad_id] * (width - len(v)))
        lens = [rl for _s, _t, rl in rows]
        if not rows:
            built, lens, queries = [[pad_id] * width], [pad_len], [(0, 1)]
            spec_of_row = [0]
        bi = torch.tensor(built, dtype=torch.long, device=dev)
        mb = {'input_ids': bi}
        if rows:
            spec_of_row = [r[0] for r in rows]
        cut = {}
        for row, pos in list(queries) + list(resps):
            sp = spec_of_row[row]
            cut[sp] = min(cut.get(sp, pos), pos)
        cuts = [int(cut.get(sp, lens[i])) for i, sp in enumerate(spec_of_row)]
        probs, resp_logp = self._forward_positions_shared(mb, lens, spec_of_row, cuts,
                                                          temperature, queries, resps)
        got = []
        for s, a in enumerate(at):
            if a is None:
                got.append(None)
                continue
            q_off, n_site, units_off, r_off, n_resp = a
            got.append((
                probs[q_off, self._gate_token_col],
                probs[q_off + 1:q_off + 1 + n_site, self._digit_cols],
                None if not units_off else {t: probs[o:o + n_site, self._digit_cols]
                                            for t, o in zip(sorted(specs[s][3]), units_off)},
                resp_logp[r_off:r_off + n_resp]))
        return got, probs


    def _forward_positions_shared(self, mb, lens, spec_of_row, cuts, temperature, queries, resps):
        self._nfwd += 1
        if self.use_ulysses_sp:
            raise RuntimeError("the numbered-label readout needs ulysses_sequence_parallel_size=1")
        ids = mb['input_ids']
        dev = ids.device
        n_rows = ids.shape[0]
        specs = sorted(set(spec_of_row))
        cut = min(cuts)
        if cut < 1 or any(int(lens[r]) <= cut for r in range(n_rows)):
            raise RuntimeError("readout cut %d does not leave a tail for every row (lengths %s); the cut has "
                               "to be the earliest position read, and every row must extend past it"
                               % (cut, [int(x) for x in lens]))
        base_of_row = torch.tensor([spec_of_row.index(sp) for sp in spec_of_row],
                                   dtype=torch.long, device=dev)
        bad = (ids[:, :cut] != ids.index_select(0, base_of_row)[:, :cut]).any(-1)
        if bool(bad.any()):
            r = int(bad.nonzero()[0, 0])
            raise RuntimeError("rows of one readout differ inside the shared prefix, so it cannot be "
                               "computed once: row %d against %d up to %d"
                               % (r, int(base_of_row[r]), cut))
        if not getattr(getattr(self.actor_module, '_fsdp_wrapped_module', self.actor_module),
                       'is_gradient_checkpointing', True):
            raise RuntimeError("the shared-prefix readout needs gradient checkpointing enabled: the prefix "
                               "forward's output is not consumed, so nothing else re-gathers its sharded "
                               "parameters for the backward. Run with GRAD_CKPT=True.")
        _ckpt = getattr(getattr(self.actor_module, '_fsdp_wrapped_module', self.actor_module),
                        '_gradient_checkpointing_func', None)
        if torch.is_grad_enabled() and getattr(_ckpt, 'keywords', {}).get('use_reentrant') is True:
            raise RuntimeError("the shared-prefix readout needs non-reentrant gradient checkpointing: "
                               "reentrant checkpointing would silently drop the prefix's gradient. Enable it "
                               "with gradient_checkpointing_kwargs={'use_reentrant': False}.")
        Capture, Frozen = _cache_classes()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            base_rows = torch.tensor([spec_of_row.index(sp) for sp in specs], dtype=torch.long, device=dev)
            reps = ids.index_select(0, base_rows)[:, :cut]
            same = torch.zeros(len(specs), dtype=torch.bool, device=dev)
            if len(specs) > 1:
                same[1:] = (reps[1:] == reps[:-1]).all(-1)
            slot_of_spec = ((~same).cumsum(0) - 1).tolist()
            p_ids = reps[~same]
            at_spec = {sp: k for k, sp in enumerate(specs)}
            take = torch.tensor([slot_of_spec[at_spec[sp]] for sp in spec_of_row],
                                dtype=torch.long, device=dev)
            captured = Capture()
            self.actor_module(input_ids=p_ids, attention_mask=None, past_key_values=captured,
                              use_cache=True, num_logits_to_keep=1)
            past = _expand_cache(captured, take, Frozen)
            t_len = max(int(lens[r]) - cut for r in range(n_rows))
            t_ids = ids[:, cut:cut + t_len].contiguous()
            out = self.actor_module(input_ids=t_ids, attention_mask=None, past_key_values=past,
                                    use_cache=False)
            logits = out.logits
            cols = self._read_cols(dev)

            def gather(pairs):
                rr = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
                pp = torch.tensor([p[1] - cut for p in pairs], dtype=torch.long, device=dev)
                own = torch.tensor([int(lens[p[0]]) - cut for p in pairs], dtype=torch.long, device=dev)
                if bool((pp < 0).any()) or bool((pp >= own).any()):
                    raise RuntimeError("a readout position falls outside the row it was cut into")
                return logits[rr, pp]

            lg = gather(queries).float() / temperature
            probs = (lg[:, cols] - lg.logsumexp(-1, keepdim=True)).exp()
            with torch.no_grad():
                if resps:
                    rlg = gather(resps).float() / temperature
                    rr = torch.tensor([p[0] for p in resps], dtype=torch.long, device=dev)
                    rp = torch.tensor([p[1] + 1 for p in resps], dtype=torch.long, device=dev)
                    tgt = ids[rr, rp].reshape(-1, 1)
                    resp_logp = (rlg.gather(-1, tgt) - rlg.logsumexp(-1, keepdim=True)).squeeze(-1)
                else:
                    resp_logp = torch.zeros(0, device=dev)
            return probs, resp_logp

    def _identity_sample(self, mini_batch, id2meta, n, prompts=None):
        ids = mini_batch['cp_row_id'].tolist()
        usable = [i for i, r in enumerate(ids)
                  if not (self._as_meta(id2meta.get(r)) or {}).get('pg_distractor')]
        groups = []
        for i in usable:
            if groups and ids[i] == ids[groups[-1][-1]]:
                groups[-1].append(i)
            else:
                groups.append([i])
        picked = []
        if groups:
            avg = max(1, sum(len(g) for g in groups) // len(groups))
            k = max(1, min(len(groups), max(int(n) // avg, int(prompts or n))))
            picked = [i for j in _LIX.spread_rows(k, len(groups)) for i in groups[j]]
        idx = torch.tensor(picked, dtype=torch.long)
        return mini_batch[idx].cuda(), [ids[i] for i in picked]


    def _cons_denom(self):
        if self._cons_passes is not None:
            return float(self._cons_passes) / self.gradient_accumulation
        return 1.0 / self.gradient_accumulation

    def _perm_seed(self, view):
        self._cons_step += 1
        table = self._label_tokens()
        return _LIX.pool_perm(random.Random(self._cons_step * _SEED_STRIDE + view),
                              self._cons.label_pool or table.space, table.space,
                              self._cons.perm_attempts)

    def _compose_action(self, p_call, tens_dist, units, numbers, n_tools):
        n_site = tens_dist.shape[0]
        vs = []
        for k in range(n_site):
            if units is None:
                parts = [tens_dist[k, numbers[i]] for i in range(n_tools)]
            else:
                parts = [tens_dist[k, numbers[i] // 10] * units[numbers[i] // 10][k, numbers[i] % 10]
                         for i in range(n_tools)]
            stacked = torch.stack(parts)
            vs.append(torch.cat([stacked, (1.0 - stacked.sum()).clamp_min(0.0).reshape(1)]))
        v = torch.stack(vs).mean(0) * p_call
        return torch.cat([v, (1.0 - p_call).reshape(1)])

    def _agreed_batches(self, n_rows_needed, rb, dev):
        nb = -(-int(n_rows_needed) // max(int(rb), 1))
        if torch.distributed.is_initialized():
            t = torch.tensor([nb], device=dev)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            nb = int(t.item())
        return nb

    def _rollout_spec_ids(self, row_ids, plen, numbers, n_tools):
        if n_tools <= 0:
            return None, 'no_tools'
        if numbers is None:
            return None, 'menu_unlabelled'
        ids, gate_pos, tens_pos, why = self._span(row_ids, plen)
        if ids is None:
            return None, why
        return (ids, gate_pos, tens_pos, sorted({numbers[i] // 10 for i in range(n_tools)})), None

    def _rollout_spec(self, resp_ids, sys, user, numbers, n_tools, tools=None):
        if n_tools <= 0:
            return None, 'no_tools'
        if numbers is None:
            return None, 'menu_unlabelled'
        txt = self._decode_text(resp_ids.tolist()) \
            if torch.is_tensor(resp_ids) else resp_ids
        ids, gate_pos, tens_pos, why = self._render_rollout(txt, sys, user, tools)
        if ids is None:
            return None, why
        return (ids, gate_pos, tens_pos, sorted({numbers[i] // 10 for i in range(n_tools)})), None

    def _read_actions(self, plan, temperature, dev, want_resp=True):
        got, handle = self._read_batch([p[0] for p in plan], temperature, dev, want_resp)
        out = [None if g is None else
               (self._compose_action(g[0], g[1], g[2], numbers, n_tools), g[3])
               for (_spec, numbers, n_tools), g in zip(plan, got)]
        return out, handle

    def _numbers_of(self, meta):
        n = int(meta['n_tools'])
        return _LIX.menu_numbers(meta['identity_of'], n), n


    def _deadband(self, kl):
        k = float(self._cons.kl_deadband or 0.0)
        out = kl
        tau_used = 0.0
        if k > 0.0:
            tau = k * float(getattr(self, '_floor', 0.0) or 0.0)
            if tau > 0.0:
                tau_used = tau
                out = torch.clamp(kl - tau, min=0.0)
        self._tau_used = tau_used
        hi = float(getattr(self._cons, 'kl_upper', 0.0) or 0.0)
        if hi > 0.0:
            mult = float(self._cons.kl_upper_coef or 0.0)
            base = torch.clamp(torch.clamp(kl, max=hi) - tau_used, min=0.0)
            out = base + mult * torch.clamp(kl - hi, min=0.0)
        return out

    def _consistency_identity_perm(self, sel, n_perm, coef, slots, temperature, dev):
        n_slots = max(int(slots), 1)
        rb = max(int(self._cons.readout_batch), 1)
        batches = [(lo, min(lo + rb, n_slots)) for lo in range(0, n_slots, rb)]
        kl_sum, other_sum, n_fire, n_other = 0.0, 0.0, 0, 0
        skips = {}
        table = self._label_tokens()
        nfwd0 = self._nfwd
        rows = []
        anchor = [None] * n_slots
        for s in range(n_slots):
            row_ids = meta = None
            plen = 0
            if s < len(sel):
                row_ids, plen, meta = sel[s]
            if meta is None:
                rows.append(None)
                continue
            numbers, n_tools = self._numbers_of(meta)
            spec, why = self._rollout_spec_ids(row_ids.tolist(), plen, numbers, n_tools)
            if spec is None:
                skips[why] = skips.get(why, 0) + 1
                rows.append(None)
                continue
            rows.append((row_ids, meta, numbers, n_tools, spec))
        rows.sort(key=lambda r: r[4][1] if r is not None else -1)
        for lo, hi in batches:
            plan = [(None, [], 0) if rows[s] is None else (rows[s][4], rows[s][2], rows[s][3])
                    for s in range(lo, hi)]
            with torch.no_grad():
                got, _handle = self._read_actions(plan, temperature, dev,
                                                  want_resp=self._cons.profile)
            anchor[lo:hi] = got
        n_use = sum(1 for r in rows if r is not None)
        for v in range(n_perm):
            perm = self._perm_seed(v)
            lut = torch.tensor(perm, dtype=torch.long, device=dev)
            view_kl, view_n, n_back = 0.0, 0, 0
            for lo, hi in batches:
                plan = []
                for s in range(lo, hi):
                    if rows[s] is None:
                        plan.append((None, [], 0))
                        continue
                    row_ids, meta, numbers, n_tools, aspec = rows[s]
                    avids = torch.tensor(aspec[0], dtype=torch.long, device=dev)
                    vids = _relabel_by_lut(table, avids.unsqueeze(0), lut)[0].tolist()
                    vnum = [perm[m] for m in numbers]
                    spec = (vids, aspec[1], aspec[2],
                            sorted({vnum[i] // 10 for i in range(n_tools)}))
                    plan.append((spec, vnum, n_tools))
                got, handle = self._read_actions(plan, temperature, dev,
                                                 want_resp=self._cons.profile)
                terms = []
                for i, b in enumerate(got):
                    if b is None or anchor[lo + i] is None:
                        continue
                    a_vec, a_logp = anchor[lo + i]
                    b_vec, b_logp = b
                    t_kl, t_gate, t_label, t_call, t_call0 = _IDK.kl_gated_parts(b_vec, a_vec.detach())
                    terms.append(t_kl)
                    if a_logp.numel() != b_logp.numel():
                        raise RuntimeError("relabelling changed the readout length, %d against %d"
                                           % (a_logp.numel(), b_logp.numel()))
                    k = a_logp.numel()
                    if k > 0:
                        with torch.no_grad():
                            other_sum += float(_IDK.k3(a_logp[-k:] - b_logp[-k:],
                                                       -_LOG_KL_CLAMP, _LOG_KL_CLAMP).mean())
                            n_other += 1
                part = handle.sum() * 0.0
                if terms:
                    kl = torch.stack(terms).sum()
                    view_kl += float(kl.detach())
                    view_n += len(terms)
                    part = part + coef * kl / max(n_use, 1) / n_perm
                (part * self._cons_denom()).backward()
                n_back += 1
            assert n_back == len(batches), \
                "%d backwards for %d readout batches; the number of graphs a backward traverses must come " \
                "from config, not the data" % (n_back, len(batches))
            if view_n:
                kl_sum += view_kl / view_n
                n_fire += 1
        assert self._nfwd - nfwd0 == len(batches) * (1 + n_perm), \
            "%d forwards for %d readout batches over %d slots and %d relabellings; the count must come " \
            "from config, not the data" % (self._nfwd - nfwd0, len(batches), n_slots, n_perm)
        item = kl_sum / max(n_fire, 1)
        self._perm_kl_acc = getattr(self, '_perm_kl_acc', 0.0) + float(item)
        self._perm_kl_n = getattr(self, '_perm_kl_n', 0) + 1
        return item

    def _consistency_identity_distractor(self, sel, coef, slots, temperature, dev):
        n_slots = max(int(slots), 1)
        rb = max(int(self._cons.readout_batch), 1)
        batches = [(lo, min(lo + rb, n_slots)) for lo in range(0, n_slots, rb)]
        max_tools = self._cons.max_tools
        kl_sum, applied_sum, n_pairs = 0.0, 0.0, 0
        d_gate_sum, d_label_sum, d_call_sum, d_call0_sum = 0.0, 0.0, 0.0, 0.0
        d_each = {}
        clone, n_read = 0.0, 0
        skips, handles = {}, []
        nfwd0 = self._nfwd
        for lo, hi in batches:
            plan, keep = [], []
            for s in range(lo, hi):
                row = sel[s] if s < len(sel) else None
                if row is None:
                    plan.append((None, [], 0))
                    continue
                resp_ids, meta, midx, target, n_clean, gid, dgroup, dgn = row
                if isinstance(meta, str):
                    meta = json.loads(meta)
                ds = (meta or {}).get('distractors') or []
                skip = None
                if (meta or {}).get('pg_distractor'):
                    skip = 'is_pg_distractor'
                elif not (0 <= midx < len(ds)):
                    skip = 'no_distractor_menu'
                elif self._fc_needs_tools(meta, ds[midx].get('tools')):
                    skip = 'fc_view_without_tools'
                elif ds[midx].get('no_view'):
                    skip = 'no_view_available'
                elif n_clean <= 0:
                    skip = 'no_clean_n_tools'
                elif float(target.abs().sum()) == 0:
                    skip = 'no_clean_target'
                elif dgn <= 0:
                    skip = 'no_distractor_group'
                if skip is not None:
                    skips[skip] = skips.get(skip, 0) + 1
                    plan.append((None, [], 0))
                    continue
                dmeta = {'identity_of': ds[midx]['identity_of'], 'n_tools': n_clean}
                dnum, dn = self._numbers_of(dmeta)
                spec, why = self._rollout_spec(resp_ids, ds[midx]['sys'], ds[midx]['user'], dnum, dn,
                                               ds[midx].get('tools'))
                if spec is None:
                    skips[str(why)] = skips.get(str(why), 0) + 1
                    plan.append((None, [], 0))
                    continue
                plan.append((spec, dnum, dn))
                keep.append((s - lo, target, n_clean, dn, gid))
            got, handle = self._read_actions(plan, temperature, dev, want_resp=False)
            handles.append(handle)
            batch_pairs = []
            for i, target, n_clean, dn, gid in keep:
                bv = got[i][0]
                assert bv.numel() == dn + 2, \
                    "readout is %d wide for a %d-tool menu; the dense and padded layouts have been confused" \
                    % (bv.numel(), dn)
                q = torch.cat([bv[:n_clean], bv[n_clean].reshape(1), bv[n_clean + 1].reshape(1)])
                p = torch.cat([target[:n_clean], target[max_tools].reshape(1),
                               target[max_tools + 1].reshape(1)])
                clone += float((bv[n_clean] - target[max_tools]).clamp_min(0).detach())
                n_read += 1
                gbar = torch.cat([dgroup[:n_clean], dgroup[max_tools].reshape(1),
                                  dgroup[max_tools + 1].reshape(1)]).detach()
                batch_pairs.append((gbar + (q - q.detach()), p))
            part = handle.sum() * 0.0
            surv_b = torch.zeros((), dtype=torch.float32, device=dev)
            if batch_pairs:
                parts_b = [_IDK.kl_gated_parts(qs, tp) for qs, tp in batch_pairs]
                kls_b = torch.stack([x[0] for x in parts_b])
                for (_i, _t, _nc, _dn, gid_b), kv in zip(keep, kls_b.detach().float().tolist()):
                    d_each.setdefault(gid_b, kv)
                with torch.no_grad():
                    d_gate_sum, d_label_sum, d_call_sum, d_call0_sum = [
                        a + b for a, b in zip(
                            (d_gate_sum, d_label_sum, d_call_sum, d_call0_sum),
                            torch.stack([torch.stack(x[1:5]) for x in parts_b]).sum(0).tolist())]
                mean_b = kls_b.mean()
                hinge_b = self._deadband(kls_b)
                tau_b = float(getattr(self, '_tau_used', 0.0))
                surv_b = (kls_b.detach() > tau_b).sum().to(torch.float32)
                kl_sum += float(mean_b.detach()) * len(batch_pairs)
                applied_sum += float(hinge_b.mean().detach()) * len(batch_pairs)
                n_pairs += len(batch_pairs)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(surv_b, op=dist.ReduceOp.SUM)
            if batch_pairs:
                part = part + coef * (hinge_b.sum() / surv_b.clamp_min(1.0)) / max(len(batches), 1)
            (part * self._cons_denom()).backward()
        assert self._nfwd - nfwd0 == len(batches) and len(handles) == len(batches), \
            "%d forwards and %d graphs for %d readout batches over %d slots; both counts must come from " \
            "config, not the data" % (self._nfwd - nfwd0, len(handles), len(batches), n_slots)
        item = kl_sum / max(n_pairs, 1)
        self._distr_kl_acc = getattr(self, '_distr_kl_acc', 0.0) + float(item)
        self._distr_kl_n = getattr(self, '_distr_kl_n', 0) + 1
        if d_each:
            q = sorted(d_each.values())
        return item

    def compute_action_dists(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()
        self._prof = {}
        responses = data.batch['responses']
        row_ids = data.batch['cp_row_id']
        metas = data.non_tensor_batch['cp_meta']
        dev = responses.device
        temperature = data.meta_info['temperature']
        max_tools = self._cons.max_tools
        n_rows = responses.size(0)
        rb = max(int(self._cons.readout_batch), 1)
        out = torch.zeros(n_rows, max_tools + 2, dtype=torch.float32, device=dev)
        ntools = torch.zeros(n_rows, dtype=torch.long, device=dev)
        have_d = ('distractor_full_ids' in data.batch.keys()
                  and 'distractor_index' in data.batch.keys())
        p_end = data.batch['input_ids'].size(-1) - responses.size(-1)
        rows_cpu = _Seqs(data.batch['input_ids'], data.batch['attention_mask'], p_end)
        views_cpu = _Seqs(data.batch['distractor_full_ids'], data.batch['distractor_full_mask'],
                          p_end) if have_d else None
        dmidx = data.batch['distractor_index'].tolist() if have_d else None
        dout = torch.zeros_like(out)
        nfwd0 = self._nfwd
        parsed = [json.loads(m) if isinstance(m, str) else m for m in metas]
        want_c = []
        nt = [0] * n_rows
        for i in range(n_rows):
            meta = parsed[i]
            if (meta or {}).get('pg_distractor'):
                continue
            numbers, n = self._numbers_of(meta)
            if n > max_tools:
                continue
            ntools[i] = n
            nt[i] = n
            want_c.append((i, numbers, n))
        specs_c = []
        for i, numbers, n in want_c:
            spec, _why = self._rollout_spec_ids(*rows_cpu[i], numbers, n)
            key = spec[1] if spec is not None else 0
            specs_c.append((key, i, spec, numbers, n))
        specs_c.sort(key=lambda t: t[0])
        nb_c = self._agreed_batches(len(want_c), rb, dev)
        nb_d = 0
        for b in range(nb_c):
            chunk = specs_c[b * rb:(b + 1) * rb]
            dest = [i for _k, i, _s, _num, _n in chunk]
            plan = [(spec, numbers, n) for _k, _i, spec, numbers, n in chunk]
            with torch.no_grad():
                got, _handle = self._read_actions(plan, temperature, dev, want_resp=False)
            for j, a in enumerate(got):
                if a is None:
                    continue
                i, n = dest[j], plan[j][2]
                out[i, :n] = a[0][:n].detach()
                out[i, max_tools] = a[0][n].detach()
                out[i, max_tools + 1] = a[0][n + 1].detach()
        if have_d:
            want_d = []
            for i in range(n_rows):
                meta = parsed[i]
                ds = (meta or {}).get('distractors') or []
                n = nt[i]
                midx = int(dmidx[i])
                if n <= 0 or n > max_tools or midx >= len(ds) or meta.get('pg_distractor') \
                        or (0 <= midx < len(ds) and ds[midx].get('no_view')):
                    continue
                want_d.append((i, meta, ds, n, midx))
            nb_d = self._agreed_batches(len(want_d), rb, dev)
            specs_d = []
            for _i, _meta, ds, n, midx in want_d:
                dnum, dn = self._numbers_of({'identity_of': ds[midx]['identity_of'], 'n_tools': n})
                sp, _why = self._rollout_spec_ids(*views_cpu[_i], dnum, dn)
                specs_d.append((sp[1] if sp is not None else 0, _i, sp, dnum, dn))
            specs_d.sort(key=lambda t: t[0])
            for b in range(nb_d):
                chunk = specs_d[b * rb:(b + 1) * rb]
                dest = [i for _k, i, _s, _num, _n in chunk]
                plan = [(sp, dnum, dn) for _k, _i, sp, dnum, dn in chunk]
                with torch.no_grad():
                    got, _handle = self._read_actions(plan, temperature, dev, want_resp=False)
                for j, a in enumerate(got):
                    if a is None:
                        continue
                    i, n = dest[j], plan[j][2]
                    dout[i, :n] = a[0][:n].detach()
                    dout[i, max_tools] = a[0][n].detach()
                    dout[i, max_tools + 1] = a[0][n + 1].detach()
        want = nb_c + nb_d
        assert self._nfwd - nfwd0 == want, \
            "%d forwards where every rank agreed on %d; the count has to be the all-reduced one or the " \
            "group stalls on a parameter all-gather" % (self._nfwd - nfwd0, want)
        self.actor_module.train()
        return out, ntools, row_ids, dout


    def _optimizer_step(self, optimizer=None):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        (optimizer if optimizer is not None else self.actor_optimizer).step()
        return grad_norm


    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)



        log_probs_lst = []
        for m, micro_batch in enumerate(micro_batches):
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature,
                                                        need_entropy=False)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs


    def _probe(self):
        every = self._cons.grad_probe_every
        if not every:
            return None
        self._upd_calls = getattr(self, '_upd_calls', -1) + 1
        if self._upd_calls % every:
            return None
        if self._grad_probe is None:
            self._grad_probe = _GP.GradProbe(self.actor_module.parameters(), reference=None)
        return self._grad_probe


    @staticmethod
    def _spread_rows(n, total):
        return _LIX.spread_rows(n, total)


    def _mode_identity(self, mb, log_prob, response_mask, id2meta, rows, K, temperature, probe):
        C = self._cons
        dev = mb['responses'].device
        n_perm_slots = C.identity_prompts or C.identity_slots
        n_p = min(n_perm_slots, len(rows))
        item = 0.0
        if C.identity_perm_coef > 0:
            seqs = _Seqs(mb['input_ids'], mb['attention_mask'],
                         mb['input_ids'].shape[-1] - mb['responses'].shape[-1])
            first = {}
            for i in range(len(rows)):
                first.setdefault(rows[i], i)
            heads = list(first.values())
            sel_p = [(seqs.row(i), seqs[i][1], id2meta.get(rows[i]))
                     for i in [heads[k] for k in self._spread_rows(min(n_p, len(heads)), len(heads))]]
            item += self._consistency_identity_perm(sel_p,
                                                    C.n_perm, C.identity_perm_coef, n_perm_slots,
                                                    temperature, dev)
        if C.identity_distractor_coef > 0 and 'distractor_responses' in mb \
                and 'clean_action_dist' in mb:
            dresp, dmidx = mb['distractor_responses'], mb['distractor_index'].tolist()
            dtgt, dn = mb['clean_action_dist'], mb['clean_n_tools']
            dgrp, dgn = mb['distractor_group_dist'], mb['distractor_group_n'].tolist()
            usable = [i for i in range(len(rows))
                      if not (self._as_meta(id2meta.get(rows[i])) or {}).get('pg_distractor')]
            ugroups = []
            for i in usable:
                if ugroups and rows[i] == rows[ugroups[-1][-1]]:
                    ugroups[-1].append(i)
                else:
                    ugroups.append([i])
            avg_u = max(1, sum(len(g) for g in ugroups) // max(len(ugroups), 1))
            kg = max(1, min(len(ugroups), int(C.identity_slots) // avg_u))
            picked = [i for j in self._spread_rows(kg, len(ugroups)) for i in ugroups[j]]
            sel = [(dresp[i], id2meta.get(rows[i]), int(dmidx[i]), dtgt[i], int(dn[i]), rows[i],
                    dgrp[i], int(dgn[i]))
                   for i in picked]
            item += self._consistency_identity_distractor(sel, C.identity_distractor_coef,
                                                          C.identity_slots, temperature, dev)
        return item

    def _check_forward_count(self, nfwd0, dev):
        if not (dist.is_available() and dist.is_initialized()):
            return
        c = torch.tensor([float(self._nfwd - nfwd0)] * 2, device=dev)
        dist.all_reduce(c[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(c[1:], op=dist.ReduceOp.MAX)
        if float(c[0]) != float(c[1]):
            raise RuntimeError(
                "forward count diverged across ranks in one micro-batch: min %d, max %d, this rank %d. "
                "Every slot must spend a forward, fired or not." % (int(c[0]), int(c[1]),
                                                                    self._nfwd - nfwd0))

    def update_policy(self, data: DataProto):
        self.actor_module.train()
        self._prof = {}

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']
        C = self._cons
        probe = self._probe()

        self._floor = float(data.meta_info.get('clean_vs_clean_floor', 0.0) or 0.0)
        id2meta = None
        if C.any_coef > 0 and self.tokenizer is not None \
                and 'cp_meta' in data.non_tensor_batch and 'cp_row_id' in data.batch.keys():
            ids_full = data.batch['cp_row_id']
            metas_full = data.non_tensor_batch['cp_meta']
            id2meta = dict(zip(ids_full.tolist(), metas_full))

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if id2meta is not None:
            select_keys.append('cp_row_id')
        for k in ('distractor_responses', 'distractor_index', 'clean_action_dist', 'clean_n_tools',
                  'distractor_group_dist', 'distractor_group_n'):
            if k in data.batch.keys():
                select_keys.append(k)
        batch = data.select(batch_keys=select_keys).batch

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, mini_batch in enumerate(dataloader):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for mb_idx, mb in enumerate(micro_batches):
                mb = mb.cuda()
                responses = mb['responses']
                response_length = responses.size(1)
                response_mask = mb['attention_mask'][:, -response_length:]

                entropy, log_prob = self._forward_micro_batch(micro_batch=mb, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=mb['old_log_probs'], log_prob=log_prob, advantages=mb['advantages'],
                    eos_mask=response_mask, cliprange=self.config.clip_ratio)
                entropy_loss = verl_F.masked_mean(entropy, response_mask)
                policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                if self.config.use_kl_loss:
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=mb['ref_log_prob'],
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef



                (policy_loss * self._cons_denom()).backward()

                self._cons_metrics = {}
                cons_item = 0.0

                mb_metrics = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, mb_metrics)

            nfwd0 = self._nfwd
            self._cons_metrics = {}
            self._cons_passes = len(micro_batches)
            try:
                sample, rows = self._identity_sample(
                    mini_batch, id2meta, C.identity_slots,
                    prompts=C.identity_prompts or C.identity_slots)
                cons_item = self._mode_identity(sample, None, None, id2meta, rows, len(rows),
                                                temperature, probe if batch_idx == 0 else None)
            finally:
                self._cons_passes = None
            self._check_forward_count(nfwd0, sample['responses'].device)
            append_to_dict(metrics, {'actor/consistency_loss': cons_item, **self._cons_metrics})

            grad_norm = self._optimizer_step()
            step_metrics = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, step_metrics)
        if getattr(self, '_distr_kl_n', 0):
            self._distr_kl_prev = self._distr_kl_acc / self._distr_kl_n
            self._distr_kl_acc, self._distr_kl_n = 0.0, 0
        if getattr(self, '_perm_kl_n', 0):
            self._perm_kl_prev = self._perm_kl_acc / self._perm_kl_n
            self._perm_kl_acc, self._perm_kl_n = 0.0, 0
        self.actor_optimizer.zero_grad()
        append_to_dict(metrics, self.prof_metrics())
        return metrics
