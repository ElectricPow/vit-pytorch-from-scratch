from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # 可复现性与数据
    seed: int = 42
    data_dir: str = "data"
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 0
    val_ratio: float = 0.1

    # 模型
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    num_layers: int = 6
    embed_dim: int = 192
    num_heads: int = 3
    mlp_dim: int = 768
    embed_dropout: float = 0.1
    attention_dropout: float = 0.0
    projection_dropout: float = 0.1
    mlp_dropout: float = 0.1
    classifier: str = "cls"  # "token" or "cls"

    # 训练
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    min_learning_rate: float = 1e-6

    # 输出
    checkpoint_dir: str = "checkpoints"


def get_config() -> Config:
    return Config()