import torch

from src.data import build_cifar10_dataloaders


train_loader, val_loader, test_loader = build_cifar10_dataloaders(
    data_dir="data",
    batch_size=128,
    eval_batch_size=256,
    num_workers=0,
    seed=42,
)

images, labels = next(iter(train_loader))

print(images.shape)
print(images.dtype)
print(labels.shape)
print(labels.dtype)
print(labels.min().item(), labels.max().item())

assert images.shape == (128, 3, 32, 32)
assert images.dtype == torch.float32
assert labels.shape == (128,)
assert labels.dtype == torch.int64
assert 0 <= labels.min().item()
assert labels.max().item() <= 9
assert torch.isfinite(images).all()