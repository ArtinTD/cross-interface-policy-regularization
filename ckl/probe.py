import torch


def pack(probes, pad_id=0):
    lens = [len(p["ids"]) for p in probes]
    out = torch.full((len(lens), max(lens)), pad_id, dtype=torch.long)
    for i, p in enumerate(probes):
        out[i, :lens[i]] = torch.tensor(p["ids"], dtype=torch.long)
    return out, lens
