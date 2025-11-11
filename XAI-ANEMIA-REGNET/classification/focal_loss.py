import torch
import torch.nn.functional as F


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):

    probs = torch.softmax(inputs, dim=1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-8)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
