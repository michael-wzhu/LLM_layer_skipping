import torch
import torch.nn.functional as F


def stable_log_softmax(p_logit, dim=-1):
    p_logit_1 = p_logit - torch.max(p_logit)
    return F.log_softmax(p_logit_1, dim=dim)


def kl_distance(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (stable_log_softmax(p_logit, dim=-1)
                                  - stable_log_softmax(q_logit, dim=-1)), -1)

    return torch.mean(_kl)


def symmetric_kl_distance(p_logit, q_logit):

    return 0.5 * (kl_distance(p_logit, q_logit) + kl_distance(q_logit, p_logit))




def jensen_shannon_divergence(x, y):

    entropy = 0.5 * - torch.sum(x * torch.log((x + y) / 2), dim=-1) + \
              0.5 * - torch.sum(y * torch.log((x + y) / 2), dim=-1)

    return torch.mean(entropy)


def soft_ce_distance(x, y):
    ce_dist = - torch.sum(x * torch.log(y), dim=-1)

    return ce_dist