import torch
from torch import nn

from configs.cifar10 import get_config
from src.data import build_cifar10_dataloaders
from src.engine import evaluate_one_epoch
from train import create_model


def main() -> None:
    config = get_config()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, _, test_loader = build_cifar10_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    model = create_model(config).to(device)
    checkpoint = torch.load(
        "checkpoints/best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"test_loss={metrics['loss']:.4f}")
    print(f"test_accuracy={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()