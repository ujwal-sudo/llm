import torch


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)


def accuracy(loader, model, device, num_batches=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(loader):
            if num_batches is not None and index >= num_batches:
                break
            predictions = model(inputs.to(device))[:, -1, :].argmax(dim=-1)
            correct += (predictions == targets.to(device)).sum().item()
            total += targets.shape[0]
    return correct / total if total else float("nan")
