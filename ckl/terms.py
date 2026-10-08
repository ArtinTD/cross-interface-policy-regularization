import torch

POOL = 100
DIGITS = 10
_EPS = 1e-12


def label_dist(tens_logits, units_logits):
    if tens_logits.shape[-1] != DIGITS or units_logits.shape[-2:] != (DIGITS, DIGITS):
        raise ValueError("expected tens (..., %d) and units (..., %d, %d), got %s and %s"
                         % (DIGITS, DIGITS, DIGITS, tuple(tens_logits.shape), tuple(units_logits.shape)))
    p_tens = torch.softmax(tens_logits.float(), dim=-1)
    p_units = torch.softmax(units_logits.float(), dim=-1)
    return (p_tens.unsqueeze(-1) * p_units).flatten(-2)


def compose_action(p_call, pool_dist, numbers):
    idx = torch.as_tensor(numbers, dtype=torch.long, device=pool_dist.device)
    if idx.numel() and (idx.min() < 0 or idx.max() >= POOL):
        raise ValueError("label numbers must lie in [0, %d): got %s" % (POOL, numbers))
    if idx.numel() != len(set(int(x) for x in numbers)):
        raise ValueError("a naming assigns DISTINCT labels to positions: %s" % (numbers,))
    declared = pool_dist.index_select(-1, idx)
    undeclared = (1.0 - declared.sum(-1, keepdim=True)).clamp_min(0.0)
    called = torch.cat([declared, undeclared], dim=-1) * p_call.unsqueeze(-1)
    return torch.cat([called, (1.0 - p_call).unsqueeze(-1)], dim=-1)


def kl(p, q_ref):
    p = p.float()
    q_ref = q_ref.float().detach()
    return (p * (torch.log(p.clamp_min(_EPS)) - torch.log(q_ref.clamp_min(_EPS)))).sum(-1)


def kl_parts(p, q_ref):
    p, q_ref = p.float(), q_ref.float().detach()
    c_p = p[..., :-1].sum(-1).clamp_min(_EPS)
    c_q = q_ref[..., :-1].sum(-1).clamp_min(_EPS)
    gate = (c_p * (torch.log(c_p) - torch.log(c_q))
            + (1 - c_p).clamp_min(_EPS) * (torch.log((1 - c_p).clamp_min(_EPS))
                                           - torch.log((1 - c_q).clamp_min(_EPS))))
    u_p = p[..., :-1] / c_p.unsqueeze(-1)
    u_q = q_ref[..., :-1] / c_q.unsqueeze(-1)
    label = kl(u_p, u_q)
    return gate + c_p.detach() * label, gate, label, c_p, c_q


def relabelling_kl(q_relabelled, q_original):
    total, gate, label, c, c_ref = kl_parts(q_relabelled, q_original)
    with torch.no_grad():
        m = {"rel_kl": float(total.mean()), "rel_gate": float(gate.mean()),
             "rel_label": float(label.mean()), "rel_p_call": float(c.mean()),
             "rel_p_call_ref": float(c_ref.mean())}
    return total.mean(), m


def twin_kl_from_means(live_means, clean_means):
    total, gate, label, c, c_ref = kl_parts(live_means, clean_means)
    with torch.no_grad():
        m = {"add_kl": float(total.mean()), "add_gate": float(gate.mean()),
             "add_label": float(label.mean()), "add_p_call": float(c.mean()),
             "add_p_call_ref": float(c_ref.mean())}
    return total.mean(), m


def twin_mean_coeff(live_means, clean_means):
    m = live_means.detach().clone().requires_grad_(True)
    val, metrics = twin_kl_from_means(m, clean_means.detach())
    coeff, = torch.autograd.grad(val, m)
    return float(val.detach()), metrics, coeff.detach()


def twin_surrogate(q_rows, coeff, group_size):
    out = None
    for row in q_rows:
        part = (coeff * row).sum() / float(max(group_size, 1))
        out = part if out is None else out + part
    return out
