from . import rows
from .label_index import rollout_span

ANCHOR, RELABELLED, CONT_ORIG, CONT_REL = 0, 1, 2, 3
PAD = -1

MAX_MENU = 48


def draw_perms(rng, space, pool=100, attempts=64, args=True):
    func = rows.pool_perm(rng, pool, space, attempts)
    arg = rows.pool_perm(rng, pool, space, attempts) if args else None
    return func, arg


def build_plain(row_ids, plen, numbers, table, span, width=16, read_id=0):
    ids, gate_pos, tens_pos, why = rollout_span(row_ids, plen, span.gate_id, span.close_id,
                                                span.think_ids, span.probe_ids, table)
    if ids is None:
        return None, why
    def row(role, ids_, cont_digit):
        return {"ids": list(ids_), "plen": plen, "read_id": read_id, "role": role,
                "cont_digit": cont_digit, "gate_pos": gate_pos, "tens_pos": list(tens_pos),
                "numbers": list(numbers)}

    block = [row(ANCHOR, ids, -1)]
    if width is not None and len(block) > width:
        return None, "block_wider_than_%d" % width
    return block, None


def pad_row():
    return {"ids": [0, 0], "plen": 1, "read_id": -1, "role": PAD, "cont_digit": -1, "gate_pos": 0,
            "tens_pos": [], "numbers": []}


def pack_blocks(units, width):
    blocks, skipped, cur = [], [], []

    def close():
        nonlocal cur
        if cur:
            blocks.append(cur + [pad_row() for _ in range(width - len(cur))])
        cur = []

    for rowset in units:
        if len(rowset) > width:
            skipped.append(rowset)
            continue
        if len(cur) + len(rowset) > width:
            close()
        cur += rowset
    close()
    return blocks, skipped


def build(row_ids, plen, numbers, table, span, func_perm, arg_perm=None, width=16, read_id=0):
    ids, gate_pos, tens_pos, why = rollout_span(row_ids, plen, span.gate_id, span.close_id,
                                                span.think_ids, span.probe_ids, table)
    if ids is None:
        return None, why
    plan = rows.plan(ids, gate_pos, tens_pos, numbers, table, func_perm, arg_perm)

    def row(role, ids_, cont_digit, nums):
        return {"ids": list(ids_), "plen": plen, "read_id": read_id, "role": role,
                "cont_digit": cont_digit, "gate_pos": gate_pos, "tens_pos": list(tens_pos),
                "numbers": list(nums)}

    block = [row(ANCHOR, ids, -1, plan["numbers"]["orig"])]
    for role_name, ids_ in plan["extra"]:
        if role_name == "rel":
            block.append(row(RELABELLED, ids_, -1, plan["numbers"]["rel"]))
    if width is not None and len(block) > width:
        return None, "block_wider_than_%d" % width
    return block, None


def interleave(long, short):
    if not short:
        return list(long)
    if not long:
        return list(short)
    total = len(long) + len(short)
    at = {min(total - 1, int((m + 0.5) * total / len(short))) for m in range(len(short))}
    out, li, si = [], 0, 0
    for i in range(total):
        if si < len(short) and (i in at or li >= len(long)):
            out.append(short[si])
            si += 1
        else:
            out.append(long[li])
            li += 1
    return out


def block_order(seq_lens, width, k, partition_fn, reading_blocks=()):
    n_blocks, rem = divmod(len(seq_lens), width)
    if rem:
        return list(range(len(seq_lens)))
    reads = [b for b in dict.fromkeys(int(b) for b in reading_blocks) if 0 <= b < n_blocks]
    rest = [b for b in range(n_blocks) if b not in set(reads)]
    if reads:
        t = n_blocks % k
        h = len(reads) - ((len(reads) - t) % k) if len(reads) >= t else 0
        if h < 0 or (n_blocks - h) % k:
            h = 0
        reads, rest = reads[:h], sorted(rest + reads[h:])
    per_block = [int(sum(seq_lens[b * width:(b + 1) * width])) for b in rest]
    parts = [[rest[i] for i in part] for part in partition_fn(per_block, k)] if rest \
        else [[] for _ in range(k)]
    for j in range(len(parts)):
        parts[j] = interleave(parts[j], reads[j::k])
    return [r for part in parts for b in part for r in range(b * width, (b + 1) * width)]
