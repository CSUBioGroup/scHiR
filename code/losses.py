import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F


def log_zinb_positive(x, mu, theta, logit_pi, eps=1e-8):
    softplus_pi = F.softplus(-logit_pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -logit_pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta.clamp(min=eps))
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return -res.sum(dim=-1)


def rna_loss_fn(x, pred_rate, pred_theta, pred_pi, l_mu, l_var, local_l_mu, local_l_var):
    recon_loss = log_zinb_positive(x, pred_rate, pred_theta, pred_pi)
    l_var = torch.clamp(l_var, min=1e-6)
    local_l_var = torch.clamp(local_l_var, min=1e-6)

    kl_divergence_l = kl(
        Normal(l_mu, torch.sqrt(l_var)),
        Normal(local_l_mu, torch.sqrt(local_l_var)),
    ).sum(dim=1)

    return torch.mean(recon_loss + kl_divergence_l)


def hic_loss_fn(x_list, recon_list):
    """
    x_list: List[Tensor], len=n_chr, each (B, 1, H, W)
    recon_list: List[Tensor], len=n_chr, each (B, 1, H, W)
    """
    assert isinstance(x_list, list)
    assert isinstance(recon_list, list)
    assert len(x_list) == len(recon_list)

    total_loss = 0.0
    for x, recon_x in zip(x_list, recon_list):
        target = torch.clamp(x, min=0.0)
        rate = F.softplus(recon_x) + 1e-8
        recon = F.poisson_nll_loss(rate, target, log_input=False, reduction="none")
        per_sample = recon.view(recon.size(0), -1).sum(dim=1)
        total_loss += per_sample.mean()
    total_loss = total_loss / len(x_list)
    return total_loss
