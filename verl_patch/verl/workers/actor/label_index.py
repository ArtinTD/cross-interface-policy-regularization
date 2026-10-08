

def pool_perm(rng, pool, space, attempts):
    base = list(range(pool))
    g = list(base)
    for _ in range(attempts):
        rng.shuffle(g)
        if g != base:
            break
    if g == base:
        raise RuntimeError("drew the identity as a relabelling %d times running" % attempts)
    return g + list(range(pool, space))


def spread_rows(n, total):
    if total <= 0 or n <= 0:
        return []
    if total <= n:
        return list(range(total))
    return [i * total // n for i in range(n)]


def menu_numbers(identity_of, n_tools):
    numbers = [None] * n_tools
    for m, i in enumerate(identity_of):
        if 0 <= i < n_tools:
            numbers[i] = m
    if any(m is None for m in numbers):
        return None
    return numbers


def readout_layout(n_ids, gate_pos, tens_positions, single=False, trim=False):
    row_len = n_ids + sum(1 for q in tens_positions if q >= n_ids)
    want = [gate_pos] + [q - 1 for q in tens_positions] + list(tens_positions)
    if any(p < 0 or p >= row_len for p in want):
        raise ValueError("readout position outside a row of %d: %s" % (row_len, want))
    return (max(want) + 1 if trim else row_len), want


def readout_batch(specs, want_resp=True, single=False):
    rows, queries, resps, at = [], [], [], []
    for s, spec in enumerate(specs):
        if spec is None:
            at.append(None)
            continue
        n_ids, gate_pos, tens_positions, n_tens, own = spec
        row_len, want = readout_layout(n_ids, gate_pos, tens_positions, single=single,
                                       trim=not want_resp)
        n_site = len(tens_positions)
        base = len(rows)
        rows.append((s, -1, row_len))
        q_off = len(queries)
        for p in want[:1 + n_site]:
            queries.append((base, p))
        units = [None] * n_tens
        if 0 <= own < n_tens:
            units[own] = len(queries)
            for p in want[1 + n_site:]:
                queries.append((base, p))
        for i in range(n_tens):
            if units[i] is not None:
                continue
            r = len(rows)
            rows.append((s, i, row_len))
            units[i] = len(queries)
            for p in want[1 + n_site:]:
                queries.append((r, p))
        n_resp = max(min(n_ids - gate_pos, row_len - 1), 1) if want_resp else 0
        r_off = len(resps)
        for p in range(row_len - n_resp - 1, row_len - 1) if want_resp else ():
            resps.append((base, p))
        at.append((q_off, n_site, tuple(units), r_off, n_resp))
    return rows, queries, resps, at


def digit_row(ids, tens_positions, digit, digit_lo):
    v = list(ids)
    for q in tens_positions:
        if q < len(v):
            v[q] = digit_lo + digit
        else:
            v.append(digit_lo + digit)
    return v


def own_tens(ids, tens_positions, needed, digit_lo):
    if not tens_positions or any(q >= len(ids) for q in tens_positions):
        return -1
    digits = {ids[q] - digit_lo for q in tens_positions}
    if len(digits) != 1:
        return -1
    d = digits.pop()
    return needed.index(d) if d in needed else -1


def label_sites(table, ids, start=0):
    out = []
    if getattr(table, 'width', 4) == 3:
        lab_of = table.label_of
        for p in range(max(start, 0), len(ids) - 2):
            if ids[p + 1] != table.und:
                continue
            lab = lab_of.get(ids[p + 2])
            if lab is None:
                continue
            if ids[p] in table.func:
                out.append((p + 2, False, lab))
            elif ids[p] in table.arg:
                out.append((p + 2, True, lab))
        return out
    for p in range(max(start, 0), len(ids) - 3):
        if ids[p + 1] != table.und:
            continue
        t, u = ids[p + 2], ids[p + 3]
        if not (table.digit_lo <= t <= table.digit_hi and table.digit_lo <= u <= table.digit_hi):
            continue
        label = (t - table.digit_lo) * 10 + (u - table.digit_lo)
        if ids[p] in table.func:
            out.append((p + 2, False, label))
        elif ids[p] in table.arg:
            out.append((p + 2, True, label))
    return out


def rollout_span(ids, plen, gate_id, close_id, think_ids, probe_ids, table):
    if plen < 1 or plen >= len(ids):
        return None, None, None, 'empty_response'
    gate_at = next((p for p in range(plen, len(ids)) if ids[p] == gate_id), -1)
    if gate_at >= 0:
        end = next((p for p in range(gate_at + 1, len(ids)) if ids[p] == close_id), -1)
        if end < 0:
            return None, None, None, 'unclosed_call'
        out = list(ids[:end + 1])
        tens = [q for q, is_arg, _lab in label_sites(table, out, gate_at - 1)
                if not is_arg and q > gate_at]
        if not tens:
            return None, None, None, 'no_label_in_call'
    else:
        keep = len(ids)
        if think_ids:
            n = len(think_ids)
            for p in range(plen, len(ids) - n + 1):
                if tuple(ids[p:p + n]) == tuple(think_ids):
                    keep = p + n
                    break
        probe = list(probe_ids)
        out = list(ids[:keep]) + probe
        gate_at = keep + probe.index(gate_id)
        if out[gate_at] != gate_id:
            return None, None, None, 'gate_not_located'
        tens = [len(out)]
    if gate_at < 1:
        return None, None, None, 'gate_at_sequence_start'
    return out, gate_at - 1, tens, None


