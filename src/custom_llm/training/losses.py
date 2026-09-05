import torch.nn.functional as F


def language_modeling_loss(inputs, targets, model, device):
    logits = model(inputs.to(device))
    return F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())


def classification_loss(inputs, targets, model, device):
    logits = model(inputs.to(device))[:, -1, :]
    return F.cross_entropy(logits, targets.to(device))
