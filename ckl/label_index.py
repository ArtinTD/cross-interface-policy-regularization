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


def menu_numbers(identity_of, n_tools):
    numbers = [None] * n_tools
    for m, i in enumerate(identity_of):
        if 0 <= i < n_tools:
            numbers[i] = m
    if any(m is None for m in numbers):
        return None
    return numbers


def digit_row(ids, tens_positions, digit, digit_lo):
    v = list(ids)
    for q in tens_positions:
        if q < len(v):
            v[q] = digit_lo + digit
        else:
            v.append(digit_lo + digit)
    return v


def label_sites(table, ids, start=0):
    out = []
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


