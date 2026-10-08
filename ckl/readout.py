import torch


def project(hidden, weight, index, cols=None):
    idx = torch.as_tensor(index, dtype=torch.long, device=hidden.device)
    h = hidden.index_select(0, idx).to(weight.dtype)
    if cols is not None:
        c = torch.as_tensor(cols, dtype=torch.long, device=hidden.device)
        weight = weight.index_select(0, c)
    return (h @ weight.t()).float()
