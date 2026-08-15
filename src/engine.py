import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    model.train()

    loss_sum = 0.0
    correct_sum = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
                error_if_nonfinite=True,
            )

        optimizer.step()

        batch_size = labels.shape[0]
        loss_sum += loss.item() * batch_size
        correct_sum += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct_sum / sample_count,
    }

@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    loss_sum = 0.0
    correct_sum = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        loss_sum += loss.item() * batch_size
        correct_sum += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct_sum / sample_count,
    }