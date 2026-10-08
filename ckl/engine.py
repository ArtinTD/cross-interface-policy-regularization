import torch

from . import probe, readout, rows, terms


ANCHOR, RELABELLED, CONT_ORIG, CONT_REL = 0, 1, 2, 3

PAD_TOKENS = 64
NEED = ("ckl_read_id", "ckl_role", "ckl_cont_digit", "ckl_gate_pos", "ckl_tens_pos", "ckl_numbers",
        "ckl_unit", "ckl_side")


def _drop_pad(row):
    return [int(x) for x in row.tolist() if int(x) >= 0]


def group_rows(read_id, role, cont_digit):
    out = {}
    for i in range(read_id.numel()):
        rid = int(read_id[i])
        if rid < 0:
            continue
        g = out.setdefault(rid, {"cont_orig": {}, "cont_rel": {}})
        r = int(role[i])
        if r == ANCHOR:
            g["anchor"] = i
        elif r == RELABELLED:
            g["rel"] = i
        elif r == CONT_ORIG:
            g["cont_orig"][int(cont_digit[i])] = i
        elif r == CONT_REL:
            g["cont_rel"][int(cont_digit[i])] = i
    return out


def stack_actions(*lists):
    vecs = [v for lst in lists for v in lst]
    if not vecs:
        return [torch.stack(lst) if lst else lst for lst in lists]
    width = max(v.shape[-1] for v in vecs)

    def pad(v):
        n = width - v.shape[-1]
        return v if not n else torch.cat([v[:-2], v.new_zeros(n), v[-2:]], dim=-1)

    return [torch.stack([pad(v) for v in lst]) for lst in lists]


def _weighted(dicts):
    out, wsum = {}, {}
    for m in dicts:
        w = float(m.get("_units", 0)) or 1.0
        for k, v in m.items():
            if k == "_units":
                continue
            out[k] = out.get(k, 0.0) + w * float(v)
            wsum[k] = wsum.get(k, 0.0) + w
    return {k: out[k] / wsum[k] for k in out}


def make_engine(base_cls, registry, gate_id_fn, table_fn, head_fn, arm_fn, vendor="ckl"):
    @registry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device=["cuda", "npu"],
                       vendor=vendor)
    class CKLEngine(base_cls):
        def forward_backward_batch(self, data, loss_function, forward_only=False):
            self._ckl_denom = (0, 0) if forward_only else self._ckl_count_denominators(data)
            self._ckl_plan = None if forward_only else self._ckl_plan_units(data, loss_function)
            self._ckl_cursor = 0
            try:
                out = super().forward_backward_batch(data, loss_function, forward_only)
            finally:
                plan, done, self._ckl_plan = self._ckl_plan, self._ckl_cursor, None
            if plan is not None and isinstance(out, dict) and "metrics" in out:
                out["metrics"]["ckl_deferred"] = [float(max(0, len(plan["turns"]) - done))]
                out["metrics"]["ckl_chunks"] = [float(len(plan["turns"]))]
                out["metrics"]["ckl_turns"] = [float(plan["n"])]
                out["metrics"]["ckl_groups"] = [float(plan["n_groups"])]
            return out

        def _ckl_count_denominators(self, data):
            import torch.distributed as dist
            if "ckl_role" not in set(data.keys()):
                return 0, 0
            role, unit, side = data["ckl_role"], data["ckl_unit"], data["ckl_side"]
            n_rel = int((role.reshape(-1) == RELABELLED).sum())
            u, s = unit.reshape(-1).tolist(), side.reshape(-1).tolist()
            n_twin = len({int(a) for a, b in zip(u, s) if int(a) >= 0 and int(b) >= 0})
            t = torch.tensor([n_rel, n_twin], dtype=torch.long,
                             device=next(self.module.parameters()).device)
            if dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.get_data_parallel_group())
            return int(t[0]), int(t[1])

        def forward_step(self, micro_batch, loss_function, forward_only):
            loss, meta = super().forward_step(micro_batch, loss_function, forward_only)
            if forward_only or getattr(self, "_ckl_plan", None) is None:
                return loss, meta
            extra, m = self._ckl_one_turn(loss_function)
            if extra is not None:
                loss = loss + extra
                m["ckl_term"] = float(extra.detach())
            if m:
                meta["metrics"] = dict(meta.get("metrics") or {}, **m)
            return loss, meta


        def _ckl_plan_units(self, data, loss_function):
            if "ckl_read_id" not in set(data.keys()):
                return None
            kw = getattr(loss_function, "keywords", None) or {}
            cfg = kw.get("ckl")
            table, gate_id = table_fn(self), gate_id_fn(self)
            cols = {k: data[k] for k in NEED}
            src = data["input_ids"]
            ids_of = [t.tolist() for t in (src.unbind() if src.is_nested else src)]
            readings = group_rows(cols["ckl_read_id"], cols["ckl_role"], cols["ckl_cont_digit"])

            def member(i, rid):
                tens = _drop_pad(cols["ckl_tens_pos"][i])
                nums = _drop_pad(cols["ckl_numbers"][i])
                if not tens or not nums:
                    return None
                return {"rid": rid, "ids": ids_of[i], "gate": int(cols["ckl_gate_pos"][i]),
                        "tens": tens, "nums": nums, "digits": rows.leading_digits(nums)}

            units, dropped = {}, 0
            for rid in sorted(readings):
                g = readings[rid]
                if "anchor" not in g:
                    dropped += 1
                    continue
                uid = int(cols["ckl_unit"][g["anchor"]])
                side = int(cols["ckl_side"][g["anchor"]])
                if uid < 0:
                    dropped += 1
                    continue
                if "rel" in g:
                    anchor, rel = member(g["anchor"], rid), member(g["rel"], rid)
                    if anchor is None or rel is None:
                        dropped += 1
                        continue
                    u = units.setdefault(uid, {"kind": "rel", "frozen": [], "live": []})
                    u["frozen"].append(anchor)
                    u["live"].append(rel)
                else:
                    if side < 0:
                        dropped += 1
                        continue
                    m = member(g["anchor"], rid)
                    if m is None:
                        dropped += 1
                        continue
                    u = units.setdefault(uid, {"kind": "twin", "frozen": [], "live": []})
                    u["live" if side == 1 else "frozen"].append(m)

            twin = [units[u] for u in sorted(units)
                    if units[u]["kind"] == "twin" and units[u]["frozen"] and units[u]["live"]]
            rel = [units[u] for u in sorted(units)
                   if units[u]["kind"] == "rel" and units[u]["frozen"] and units[u]["live"]]
            dropped += sum(1 for u in units.values() if not (u["frozen"] and u["live"]))

            base = dict(G=self._ckl_group_size(cfg), table=table, gate_id=gate_id,
                        ids0=(ids_of[0][:PAD_TOKENS] if ids_of else [0] * PAD_TOKENS))
            groups = self._ckl_group_means(twin, base)
            turns = []
            for g in groups:
                for k, m in enumerate(g["live"]):
                    turns.append({"kind": "surr", "frozen": [], "live": [m], "group": g,
                                  "report": k == 0})
            turns += [dict(u, kind="rel") for u in rel]
            return dict(base, turns=turns, units=turns, n=self._ckl_agree(len(turns)),
                        n_groups=len(groups), dropped=dropped)

        def _ckl_group_means(self, twin, base):
            n = self._ckl_agree(len(twin))
            out = []
            for i in range(n):
                unit = twin[i] if i < len(twin) else None
                frozen, live, _z, _t = self._ckl_read_group(unit, base, base["G"], grad=False)
                if unit is None or not frozen or not live:
                    continue
                q_l, q_f = stack_actions([live[r] for r in sorted(live)],
                                         [frozen[r] for r in sorted(frozen)])
                val, m, coeff = terms.twin_mean_coeff(q_l.mean(0, keepdim=True),
                                                      q_f.mean(0, keepdim=True))
                m["add_members"] = float(len(live))
                m["add_members_ref"] = float(len(frozen))
                out.append({"live": unit["live"][:base["G"]], "coeff": coeff, "value": val,
                            "metrics": m, "size": len(live)})
            return out

        def _ckl_group_size(self, cfg):
            return max(1, int(getattr(cfg, "group_size", 1)))

        def _ckl_one_turn(self, loss_function):
            plan = self._ckl_plan
            i = self._ckl_cursor
            if i >= plan["n"]:
                return None, {}
            self._ckl_cursor = i + 1
            self._ckl_dropped = getattr(self, "_ckl_dropped", 0) + (plan["dropped"] if i == 0 else 0)
            unit = plan["turns"][i] if i < len(plan["turns"]) else None
            if unit is not None:
                part, m = self._ckl_group_term(unit, plan, loss_function)
                read = int(m.get("_units", 0)) if part is not None else 0
                out = dict(self._ckl_drain(read), **_weighted([m] if part is not None else []))
                return part, dict(out, ckl_pad_turn=0.0)
            return self._ckl_pad_turn(plan), dict(self._ckl_drain(0), ckl_pad_turn=1.0)

        def _ckl_drain(self, read):
            out = {"ckl_readings": float(read), "ckl_dropped": float(getattr(self, "_ckl_dropped", 0))}
            self._ckl_dropped = 0
            return out

        def _ckl_agree(self, n):
            import torch.distributed as dist
            if not dist.is_initialized():
                return n
            t = torch.tensor([n], device=next(self.module.parameters()).device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.get_data_parallel_group())
            return int(t.item())


        def _ckl_member_probes(self, m, table):
            out = [{"ids": list(m["ids"])}]
            for d in m["digits"]:
                out.append({"ids": rows.digit_row(m["ids"], m["tens"], int(d), table.digit_lo)})
            return out

        def _ckl_forward(self, probes, grad):
            dev = next(self.module.parameters()).device
            t_ids, lens = probe.pack(probes)
            t_dev = t_ids.to(dev)
            fwd = dict(input_ids=t_dev, attention_mask=None, use_cache=False, return_dict=True)
            if grad:
                self.module(**fwd)
                h, w = head_fn()
                return h, w, int(t_ids.shape[1]), lens
            arm_fn(True)
            try:
                with torch.no_grad():
                    self.module(**fwd)
                h, w = head_fn()
            finally:
                arm_fn(False)
            return h, w, int(t_ids.shape[1]), lens

        def _ckl_read_group(self, unit, plan, slots, grad=True):
            table, gate_id = plan["table"], plan["gate_id"]
            pad = [{"ids": list(plan["ids0"])}]
            out = {"frozen": {}, "live": {}}
            zeros, tokens = [], 0
            for which in ("frozen", "live"):
                members = (unit or {}).get(which, [])[:slots]
                for slot in range(slots):
                    m = members[slot] if slot < len(members) else None
                    got = self._ckl_forward(self._ckl_member_probes(m, table) if m else pad,
                                            grad=(grad and which == "live"))
                    if grad and which == "live":
                        zeros.append(self._ckl_zero(got))
                        tokens += got[2] * len(got[3])
                    if m is None:
                        continue
                    if grad and which == "live":
                        q = self._ckl_member_q(got, m, table, gate_id)
                    else:
                        with torch.no_grad():
                            q = self._ckl_member_q(got, m, table, gate_id)
                    if q is None:
                        self._ckl_dropped = getattr(self, "_ckl_dropped", 0) + 1
                    else:
                        out[which][m["rid"]] = q
            return out["frozen"], out["live"], [z for z in zeros if z is not None], tokens

        @staticmethod
        def _ckl_zero(got):
            h, w = got[0], got[1]
            if h is None or w is None:
                return None
            flat = h.reshape(-1, h.shape[-1]) if h.dim() == 3 else h
            return readout.project(flat, w, [0], [0]).sum() * 0.0

        def _ckl_member_q(self, got, m, table, gate_id):
            h, w, width, lens = got
            if h is None or w is None:
                return None
            flat = h.reshape(-1, h.shape[-1]) if h.dim() == 3 else h

            def read(row, offset, cols=None):
                if not (0 <= offset < lens[row]):
                    return None
                return readout.project(flat, w, [row * width + offset], cols)

            digit_cols = [table.digit_lo + j for j in range(10)]
            tens, nums, digits = m["tens"], m["nums"], m["digits"]
            lg = read(0, m["gate"])
            tens_rows = [read(0, p - 1, digit_cols) for p in tens]
            if lg is None or any(x is None for x in tens_rows):
                return None
            p_call = torch.log_softmax(lg, -1)[0, gate_id].exp()
            units = torch.zeros((len(tens), 10, 10), dtype=torch.float32, device=flat.device)
            for j, d in enumerate(digits):
                got_j = [read(1 + j, p, digit_cols) for p in tens]
                if any(x is None for x in got_j):
                    return None
                units[:, int(d), :] = torch.cat(got_j, 0)
            pool = terms.label_dist(torch.cat(tens_rows, 0), units)
            return terms.compose_action(p_call, pool[0], nums)

        def _ckl_pad_turn(self, plan):
            _f, _l, zeros, _t = self._ckl_read_group(None, plan, 1)
            if not zeros:
                return None
            total = zeros[0]
            for z in zeros[1:]:
                total = total + z
            return total


        def _ckl_group_term(self, unit, plan, loss_function):
            kw = getattr(loss_function, "keywords", None) or {}
            cfg = kw.get("ckl")
            lam_rel = getattr(cfg, "lam_rel", 0.0)
            lam_twin = getattr(cfg, "lam_twin", 0.0)
            frozen, live, zeros, tokens = self._ckl_read_group(unit, plan, 1)

            total, out, units = None, {}, 0
            for z in zeros:
                total = z if total is None else total + z
            n_rel_all, n_twin_all = getattr(self, "_ckl_denom", (0, 0))

            if unit["kind"] == "rel" and lam_rel > 0:
                ids = [rid for rid in sorted(live) if rid in frozen]
                if ids:
                    rel, ref = stack_actions([live[r] for r in ids], [frozen[r].detach() for r in ids])
                    val, m = terms.relabelling_kl(rel, ref)
                    part = lam_rel * val * (len(ids) / max(n_rel_all, 1))
                    total = part if total is None else total + part
                    out.update(m)
                    out["rel_readings"] = float(len(ids))
                    units = len(ids)
            elif unit["kind"] == "surr" and lam_twin > 0 and live:
                g = unit["group"]
                part = terms.twin_surrogate([live[r] for r in sorted(live)], g["coeff"], g["size"])
                if part is not None:
                    part = lam_twin * part * (1.0 / max(n_twin_all, 1))
                    total = part if total is None else total + part
                if unit.get("report"):
                    out.update(g["metrics"])
                    units = 1

            out["ckl_live_tokens"] = float(tokens)
            out["_units"] = units
            return total, out

    return CKLEngine
