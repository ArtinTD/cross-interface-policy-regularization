import torch


def kl_categorical(q, p, eps=1e-9):
    qc, pc = q.clamp_min(0.0), p.clamp_min(0.0)
    return (qc * ((qc + eps).log() - (pc + eps).log())).sum()


def kl_gated_parts(q, p, eps=1e-9):
    c = q[:-1].sum().clamp_min(eps)
    c0 = p[:-1].sum().clamp_min(eps)
    oc = (1.0 - c).clamp_min(0.0)
    oc0 = (1.0 - c0).clamp_min(0.0)
    gate = (c * ((c + eps).log() - (c0 + eps).log())
            + oc * ((oc + eps).log() - (oc0 + eps).log()))
    label = kl_categorical(q[:-1] / c, (p[:-1] / c0).detach(), eps)
    return gate + c.detach() * label, gate, label, c, c0


def k3(d, lo, hi):
    dl = d.clamp(min=lo, max=hi)
    e = torch.exp(dl)
    return e - 1.0 - dl + (e - 1.0) * (d - dl)
