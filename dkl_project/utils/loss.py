import torch
import torch.nn.functional as F


def pinball_loss(preds, targets, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss
