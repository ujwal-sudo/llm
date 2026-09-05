import torch

from ..evaluation.metrics import accuracy
from .losses import classification_loss


def evaluate(model, loader, device, eval_iter=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(loader):
            if eval_iter is not None and index >= eval_iter:
                break
            losses.append(classification_loss(inputs, targets, model, device).item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs=5, eval_freq=50, eval_iter=5):
    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": [], "examples_seen": []}
    step = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            classification_loss(inputs, targets, model, device).backward()
            optimizer.step()
            if step % eval_freq == 0:
                history["train_losses"].append(evaluate(model, train_loader, device, eval_iter))
                history["val_losses"].append(evaluate(model, val_loader, device, eval_iter))
                history["examples_seen"].append((step + 1) * inputs.shape[0])
            step += 1
        history["train_accs"].append(accuracy(train_loader, model, device, eval_iter))
        history["val_accs"].append(accuracy(val_loader, model, device, eval_iter))
        print(f"Epoch {epoch + 1}: train_acc={history['train_accs'][-1]:.3f}, val_acc={history['val_accs'][-1]:.3f}")
    return history
