from dataclasses import dataclass

import torch


@dataclass
class CKLConfig:
    lam_rel: float = 0.0
    lam_twin: float = 0.0
    distractor_in_pg: bool = True
    group_size: int = 1


def excluded_from_pg(data, ckl):
    keys = data.keys()
    flag = None
    if "ckl_synthetic" in keys:
        flag = data["ckl_synthetic"].to(torch.bool).reshape(-1)
    if not ckl.distractor_in_pg and "ckl_interface" in keys:
        add = data["ckl_interface"].reshape(-1) == 1
        flag = add if flag is None else (flag | add)
    return flag


def mask_synthetic(data, ckl=None):
    flag = excluded_from_pg(data, ckl or CKLConfig())
    if flag is None:
        return data, 0
    if not bool(flag.any()):
        return data, 0
    mask = data["response_mask"]
    if mask.is_nested:
        rows = list(mask.unbind())
        for i in range(len(rows)):
            if bool(flag[i]):
                rows[i] = torch.zeros_like(rows[i])
        data = data.clone(False)
        data["response_mask"] = torch.nested.as_nested_tensor(rows, layout=mask.layout)
    else:
        m = mask.clone()
        m[flag] = 0
        data = data.clone(False)
        data["response_mask"] = m
    return data, int(flag.sum())


def ckl_loss(config, model_output, data, dp_group=None, ckl=None):
    ckl = ckl or CKLConfig()
    data, n_synth = mask_synthetic(data, ckl)

    live = int(data["response_mask"].sum())
    if live == 0:
        zero = torch.zeros((), dtype=torch.float32,
                           device=getattr(data["response_mask"], "device", None))
        return zero, {"ckl_rows_out_of_pg": float(n_synth), "ckl_pg_tokens": 0.0, "pg_skipped": 1.0}

    from verl.workers.utils.losses import ppo_loss

    loss, metrics = ppo_loss(config=config, model_output=model_output, data=data, dp_group=dp_group)

    metrics["ckl_rows_out_of_pg"] = float(n_synth)
    metrics["ckl_pg_tokens"] = float(live)
    metrics["pg_skipped"] = 0.0
    return loss, metrics
