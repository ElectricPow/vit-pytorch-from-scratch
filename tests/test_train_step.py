import torch
import torch.nn.functional as F

from src.transformer import VisionTransformer

def test_one_train_step_updates_parameters():
    torch.manual_seed(42)

    # 测试使用小模型
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_layers=2,
        embed_dim=64,
        num_heads=4,
        mlp_dim=128,
        classifier="cls",
        num_classes=10,
    )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    old_weight = model.head.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (4, 10)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()

    optimizer.step()

    new_weight = model.head.weight.detach()
    assert not torch.equal(old_weight, new_weight)