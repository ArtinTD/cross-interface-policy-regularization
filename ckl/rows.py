from .label_index import digit_row, label_sites


def relabel(ids, table, func_perm, arg_perm=None):
    out = list(ids)
    for pos, is_arg, label in label_sites(table, out):
        perm = arg_perm if is_arg else func_perm
        if perm is None:
            continue
        new = perm[label]
        out[pos] = table.digit_lo + new // 10
        out[pos + 1] = table.digit_lo + new % 10
    return out


def leading_digits(numbers):
    return sorted({int(n) // 10 for n in numbers})


def read_positions(gate_pos, tens_positions):
    return gate_pos, [p - 1 for p in tens_positions], list(tens_positions)


def plan(ids, gate_pos, tens_positions, numbers_orig, table, func_perm, arg_perm=None):
    numbers_rel = [func_perm[n] for n in numbers_orig]
    rel_ids = relabel(ids, table, func_perm, arg_perm)
    d_orig, d_rel = leading_digits(numbers_orig), leading_digits(numbers_rel)

    extra = [("rel", rel_ids)]
    reads = {"orig": read_positions(gate_pos, tens_positions),
             "rel": read_positions(gate_pos, tens_positions)}
    for d in d_orig:
        extra.append(("orig_cont_%d" % d, digit_row(ids, tens_positions, d, table.digit_lo)))
        reads["orig_cont_%d" % d] = read_positions(gate_pos, tens_positions)
    for d in d_rel:
        extra.append(("rel_cont_%d" % d, digit_row(rel_ids, tens_positions, d, table.digit_lo)))
        reads["rel_cont_%d" % d] = read_positions(gate_pos, tens_positions)

    return {"extra": extra, "reads": reads,
            "numbers": {"orig": numbers_orig, "rel": numbers_rel},
            "cont_digits": {"orig": d_orig, "rel": d_rel}}


