import torch

from src.data import build_cifar10_dataloaders
from src.transformer import VisionTransformer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = VisionTransformer(
    image_size=32,
    patch_size=4,
    in_channels=3,
    num_layers=6,
    embed_dim=192,
    num_heads=3,
    mlp_dim=768,
    num_classes=10,
    classifier="cls"
)
model = model.to(device)
model.eval()

train_loader, val_loader, test_loader = build_cifar10_dataloaders(
    data_dir="data",
    batch_size=128,
    eval_batch_size=256,
    num_workers=0,
    seed=42,
)

images, labels = next(iter(train_loader))
images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

with torch.no_grad():
    logits = model(images)

print(images.shape)
print(images.dtype)
print(labels.shape)
print(labels.dtype)
print(labels.min().item(), labels.max().item())
print(logits.shape)

assert images.shape == (128, 3, 32, 32)
assert images.dtype == torch.float32
assert labels.shape == (128,)
assert labels.dtype == torch.int64
assert 0 <= labels.min().item()
assert labels.max().item() <= 9
assert torch.isfinite(images).all()

assert logits.shape == (images.shape[0], 10)
assert labels.shape == (images.shape[0],)
assert torch.isfinite(logits).all()