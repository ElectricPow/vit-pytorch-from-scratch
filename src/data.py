from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def make_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),#(H,W,C) -> (C,H,W)
        #uin8 -> float32, [0,255] -> [0,1]
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    return train_transform, eval_transform

def bulid_cifar10_datasets(
        data_dir: str | Path = "data",
        val_ratio: float = 0.1,
        seed: int = 42
):
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0,1), but got {val_ratio}")

    train_transform, eval_transform = make_transforms()

    train_full = datasets.CIFAR10(
        root = data_dir,
        train = True,
        download = True,
        transform = train_transform
    )

    val_full = datasets.CIFAR10(
        root = data_dir,
        train = True,
        download = False,
        transform = eval_transform
    )

    test_dataset = datasets.CIFAR10(
        root = data_dir,
        train = False,
        download = True,
        transform = eval_transform
    )

    num_examples = len(train_full)
    val_size = int(val_ratio * num_examples)

    generator = torch.Generator().manual_seed(seed)
    # Shuffle the indices
    indices = torch.randperm(num_examples, generator=generator).tolist()

    train_indices = indices[:val_size]
    val_indices = indices[val_size:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    return train_dataset, val_dataset, test_dataset


def build_cifar10_dataloaders(
        data_dir: str | Path = "data",
        batch_size: int = 128,
        eval_batch_size: int = 256,
        num_workers: int = 0,
        val_ratio: float = 0.1,
        seed: int = 42
):
    train_set, val_set, test_set = bulid_cifar10_datasets(data_dir, val_ratio, seed)

    # CUDA is available, pin_memory should be set to True to speed up data transfer to GPU
    pin_memory = torch.cuda.is_available()

    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader, test_loader
