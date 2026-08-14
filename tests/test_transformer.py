import torch
from src.transformer import VisionTransformer


def test_VisionTransformer():
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_layers=4,
        embed_dim=64,
        num_heads=8,
        mlp_dim=256,
    )

    images = torch.randn(2, 3, 32, 32)
    tokens = model(images)

    assert tokens.shape == (2, 65, 64)
    assert tokens[:, 0].shape == (2, 64)
    assert tokens[:, 0:1].shape == (2, 1, 64)
    assert tokens[:, 1:].shape == (2, 64, 64)