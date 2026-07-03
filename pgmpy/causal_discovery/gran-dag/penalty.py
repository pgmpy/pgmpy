import torch


def compute_penalty(list, p=2, target=0.0):
    penalty = 0
    for m in list:
        penalty += torch.norm(m - target, p=p) ** p
    return penalty


def compute_group_lasso_l2_penalty(weight):
    # weight shape: (num_vars, hid, input)
    return torch.sum(torch.norm(weight, p=2, dim=1))
