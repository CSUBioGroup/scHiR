import csv
import os
import random
import re

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def save_content_to_csv(content, file):
    content = [[str(j) for j in content]]
    with open(file, "a") as f:
        writer = csv.writer(f)
        writer.writerows(content)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def safe_float(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return float(x)


def sanitize_name(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


def grad_norm_of_module(module, norm_type=2):
    total = 0.0
    has_grad = False
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += g.norm(norm_type).item() ** norm_type
            has_grad = True
    if not has_grad:
        return 0.0
    return total ** (1.0 / norm_type)


def flatten_eval_results(results):
    flat = {}
    for label_name, metric_dict in results.items():
        prefix = sanitize_name(label_name)
        for k, v in metric_dict.items():
            flat[f"{prefix}_{k}"] = safe_float(v)
    return flat
