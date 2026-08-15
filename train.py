from pathlib import Path

import torch
from torch import nn

from configs.cifar10 import get_config
from src.data import build_cifar10_dataloaders
from src.engine import evaluate_one_epoch, train_one_epoch
from src.transformer import VisionTransformer
from utils import count_parameters, save_checkpoint, seed_everything


def create_model(config) -> VisionTransformer:
    return VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        embed_dropout=config.embed_dropout,
        attention_dropout=config.attention_dropout,
        projection_dropout=config.projection_dropout,
        mlp_dropout=config.mlp_dropout,
        classifier=config.classifier,
        num_classes=config.num_classes,
    )


def main() -> None:
    config = get_config()
    seed_everything(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, _ = build_cifar10_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    model = create_model(config).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.min_learning_rate,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    best_val_accuracy = 0.0

    print(f"device: {device}")
    print(f"trainable parameters: {count_parameters(model):,}")

    for epoch in range(config.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=config.grad_clip,
        )

        val_metrics = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        print(
            f"epoch {epoch + 1:03d}/{config.num_epochs:03d} "
            f"lr={current_lr:.6g} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        is_best = val_metrics["accuracy"] > best_val_accuracy
        if is_best:
            best_val_accuracy = val_metrics["accuracy"]

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_accuracy=best_val_accuracy,
        )

        if is_best:
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_accuracy=best_val_accuracy,
            )


if __name__ == "__main__":
    main()