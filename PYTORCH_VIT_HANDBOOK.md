# PyTorch ViT 从零复现速查手册

这份手册服务于本项目的 CIFAR-10 Vision Transformer（ViT）复现。它不是完整的 PyTorch 教程，而是用于随时查阅：某个模块怎么写、张量形状如何变化、训练循环需要哪些步骤，以及从 JAX/Flax 官方实现迁移到 PyTorch 时要注意什么。

## 1. 复现目标与原则

第一版目标：

- 使用 PyTorch 手写标准 ViT 的核心结构；
- 使用 CIFAR-10 的 `32×32` RGB 图片训练和验证；
- 自己实现多头自注意力，不用 `nn.TransformerEncoderLayer` 隐藏核心逻辑；
- 能完成前向传播、反向传播、少量样本过拟合、完整训练、验证和 checkpoint；
- 先保证正确，再考虑性能、多 GPU、混合精度和预训练权重。

推荐开发原则：

1. 每完成一个模块，立刻写形状测试。
2. 先使用很小的模型和随机输入调试。
3. 先在 16～32 张真实图片上验证能够过拟合。
4. 先使用单设备、普通精度和 eager execution。
5. 不要在第一版同时加入多 GPU、`torch.compile`、混合精度和复杂增强。

## 2. 推荐项目结构

```text
vit-pytorch-from-scratch/
├── configs/
│   └── cifar10.py              # 模型与训练配置
├── src/
│   ├── patch_embedding.py      # 图片转 Patch token
│   ├── attention.py            # 手写多头自注意力
│   ├── mlp.py                  # Transformer 中的 MLP
│   ├── encoder.py              # EncoderBlock 与 Encoder
│   ├── transformer.py          # Patch、CLS、位置编码与 Encoder
│   ├── vit.py                  # 分类头（也可并入 transformer.py）
│   └── data.py                 # CIFAR-10 Dataset 与 DataLoader
├── tests/
│   ├── test_patch_embedding.py
│   ├── test_attention.py
│   ├── test_mlp.py
│   ├── test_encoder.py
│   ├── test_transformer.py
│   └── test_data.py
├── train.py                    # CIFAR-10 训练循环
├── evaluate.py                 # 验证与 checkpoint 推理
├── requirements.txt
└── README.md
```

## 3. JAX/Flax 与 PyTorch 对照

| JAX/Flax | PyTorch | 说明 |
|---|---|---|
| `flax.linen.Module` | `torch.nn.Module` | 模型模块基类 |
| `@nn.compact` | 通常不需要 | PyTorch 通常在 `__init__` 创建层 |
| Flax 的 `__call__` | PyTorch 的 `forward` | 定义前向计算 |
| `model.init(rng, x)` | 创建 `nn.Module` 对象 | PyTorch 层通常在构造时初始化参数 |
| `model.apply({"params": params}, x)` | `model(x)` | 执行前向传播 |
| `self.param(...)` | `nn.Parameter(...)` | 注册自定义可训练参数 |
| Flax 参数树 | `model.parameters()` / `state_dict()` | 管理模型参数 |
| `nn.Dense` | `nn.Linear` | 全连接层 |
| `nn.Conv` | `nn.Conv2d` | 二维卷积 |
| `jnp.reshape` | `tensor.reshape` | 改变形状 |
| `jnp.concatenate` | `torch.cat` | 拼接张量 |
| `nn.gelu` | `nn.GELU` / `F.gelu` | GELU 激活 |
| `train=True`、`deterministic=False` | `model.train()` | 训练模式 |
| `train=False`、`deterministic=True` | `model.eval()` | 验证/推理模式 |
| `jax.value_and_grad` | `loss.backward()` | 自动求梯度 |
| `optax.update` | `optimizer.step()` | 更新参数 |
| 显式 `PRNGKey` | `torch.manual_seed` 等 | 随机性管理 |
| `pmap`、`pmean` | DDP 等 | 多设备训练，第一版不使用 |

### 3.1 参数状态的主要区别

Flax 将模型结构和参数分开：

```python
variables = model.init(rng, images, train=False)
logits = model.apply(variables, images, train=False)
```

PyTorch 将参数保存在模型对象内部：

```python
model = VisionTransformer(...)
logits = model(images)
```

PyTorch 中调用 `model(images)`，不要直接调用 `model.forward(images)`；`nn.Module.__call__` 还会处理框架 hooks 等行为。

## 4. 必须牢记的张量约定

### 4.1 图像格式

官方 JAX/Flax ViT 常使用：

```text
(B, H, W, C)
```

PyTorch/torchvision 常使用：

```text
(B, C, H, W)
```

CIFAR-10 batch 示例：

```text
(128, 3, 32, 32)
```

### 4.2 Token 格式

进入 Transformer 后统一使用：

```text
(B, N, D)
```

- `B`：batch size；
- `N`：token 数量；
- `D`：embedding/hidden dimension。

### 4.3 多头 Attention 格式

常用内部排列：

```text
(B, H, N, Dh)
```

- `H`：注意力头数；
- `Dh = D / H`：每个头的维度。

Attention 分数：

```text
(B, H, N, N)
```

其中 `scores[b, h, i, j]` 表示第 `b` 张图片、第 `h` 个头中，第 `i` 个 query 对第 `j` 个 key 的分数。

### 4.4 权重排列差异

Linear：

```text
Flax Dense kernel：       (in_features, out_features)
PyTorch Linear weight：   (out_features, in_features)
```

Conv：

```text
Flax Conv kernel：        (Kh, Kw, Cin, Cout)
PyTorch Conv2d weight：   (Cout, Cin, Kh, Kw)
```

以后转换官方预训练参数时必须转置；第一版从零训练暂时不用处理。

## 5. 常用 Python 与 PyTorch 语法

### 5.1 定义模块

```python
import torch
from torch import nn


class ExampleModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

关键点：

- 必须继承 `nn.Module`；
- 必须先调用 `super().__init__()`；
- 子层通常在 `__init__` 中赋值为 `self.xxx`；
- 数据流写在 `forward` 中；
- 外部使用 `module(x)`。

### 5.2 `ModuleList`

多个需要训练的模块必须注册：

```python
self.blocks = nn.ModuleList([
    EncoderBlock(...)
    for _ in range(depth)
])
```

前向：

```python
for block in self.blocks:
    x = block(x)
```

不要使用普通 Python 列表保存需要训练的子模块，否则它们可能不会被 `model.parameters()` 和 `state_dict()` 正确发现。

### 5.3 `nn.Parameter`

用于 CLS token、位置编码等自定义参数：

```python
self.cls_token = nn.Parameter(
    torch.zeros(1, 1, embed_dim)
)

self.pos_embedding = nn.Parameter(
    torch.zeros(1, num_patches + 1, embed_dim)
)
```

普通 `torch.Tensor` 不会自动成为可训练参数；`nn.Parameter` 会被模块注册，并在反向传播时获得梯度。

### 5.4 常用形状属性

```python
b, n, d = x.shape
x.ndim
x.dtype
x.device
x.numel()
```

`numel()` 返回元素总数，类似 NumPy/JAX 的 `.size`。

### 5.5 `reshape`、`view`、`flatten`、`transpose`、`permute`

```python
x = x.reshape(b, n, num_heads, head_dim)
x = x.transpose(1, 2)           # 只交换两个维度
x = x.permute(0, 2, 1, 3)      # 按给定顺序重排全部维度
x = x.flatten(2)                # 从第 2 维开始合并
```

`transpose`/`permute` 后张量可能不是连续内存；`reshape` 通常会在必要时处理复制。若使用 `view` 遇到连续性错误，可以先：

```python
x = x.contiguous().view(...)
```

本项目优先使用更安全直观的 `reshape`。

### 5.6 拼接、扩展与广播

```python
cls = self.cls_token.expand(b, -1, -1)  # (1,1,D) → (B,1,D)
x = torch.cat((cls, x), dim=1)          # (B,1,D)+(B,N,D) → (B,N+1,D)
x = x + self.pos_embedding              # (B,N,D)+(1,N,D)，自动广播
```

`-1` 表示该维保持原长度。

`expand` 通常只创建广播视图，不真正复制全部数据；`repeat` 会实际重复数据。CLS token 通常优先用 `expand`。

### 5.7 `@` 与 `matmul`

```python
scores = q @ k.transpose(-2, -1)
output = attention @ v
```

`@` 等价于 `torch.matmul`，对最后两个维度做矩阵乘法，并广播前面的 batch/head 维度。

## 6. 常用神经网络模块

### 6.1 `nn.Linear`

```python
layer = nn.Linear(768, 3072)
x = layer(x)
```

若 `x.shape == (B, N, 768)`，输出：

```text
(B, N, 3072)
```

Linear 只改变最后一维。

参数：

```text
weight：(3072, 768)
bias：  (3072,)
```

### 6.2 `nn.Conv2d`

```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=4,
    stride=4,
)
```

输入：

```text
(B, 3, 32, 32)
```

输出：

```text
(B, 64, 8, 8)
```

### 6.3 `nn.LayerNorm`

```python
norm = nn.LayerNorm(embed_dim)
x = norm(x)
```

对最后一维归一化：

```text
(B,N,D) → (B,N,D)
```

### 6.4 `nn.Dropout`

```python
self.dropout = nn.Dropout(0.1)
x = self.dropout(x)
```

- `model.train()`：Dropout 开启；
- `model.eval()`：Dropout 关闭。

不需要像 JAX 一样显式传递 Dropout PRNGKey。

### 6.5 GELU 与 Softmax

```python
from torch.nn import functional as F

x = F.gelu(x)
weights = F.softmax(scores, dim=-1)
```

Attention 必须沿 key 维，即最后一维执行 Softmax。

## 7. 参数初始化

PyTorch 层自带默认初始化，但为了对照官方 ViT，可以显式初始化。

```python
from torch import nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.normal_(module.bias, std=1e-6)

    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

应用到所有子模块：

```python
model.apply(init_weights)
```

注意带下划线的初始化函数，例如 `xavier_uniform_`，会原地修改参数。

自定义参数：

```python
nn.init.normal_(self.pos_embedding, std=0.02)
nn.init.zeros_(self.cls_token)
```

分类头若要仿照官方实现：

```python
nn.init.zeros_(self.head.weight)
nn.init.zeros_(self.head.bias)
```

第一版也可以保留 PyTorch 默认的 Linear 初始化，但必须在 README 中记录选择。

## 8. Patch Embedding 速查

推荐接口：

```python
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)        # (B,D,Hp,Wp)
        x = x.flatten(2)             # (B,D,N)
        x = x.transpose(1, 2)        # (B,N,D)
        return x
```

CIFAR-10 示例：

```text
输入：(B,3,32,32)
Patch：4×4
输出网格：8×8
Patch 数：64
输出：(B,64,D)
```

建议增加输入检查：

```python
if images.ndim != 4:
    raise ValueError(...)

if images.shape[-2:] != (self.image_size, self.image_size):
    raise ValueError(...)
```

## 9. 手写多头自注意力速查

核心模块骨架：

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        qkv = self.qkv(x)                         # (B,N,3D)
        qkv = qkv.reshape(
            b, n, 3, self.num_heads, self.head_dim
        )                                         # (B,N,3,H,Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3,B,H,N,Dh)
        q, k, v = qkv.unbind(dim=0)               # 每个都是 (B,H,N,Dh)

        scores = (q @ k.transpose(-2, -1)) * self.scale
                                                    # (B,H,N,N)
        attention = scores.softmax(dim=-1)          # (B,H,N,N)
        attention = self.attention_dropout(attention)

        x = attention @ v                           # (B,H,N,Dh)
        x = x.transpose(1, 2).reshape(b, n, d)      # (B,N,D)
        x = self.proj(x)                            # (B,N,D)
        x = self.projection_dropout(x)
        return x
```

关键检查：

```python
assert embed_dim % num_heads == 0
assert output.shape == input.shape
```

若临时返回 Attention 权重做测试：

```python
row_sums = attention.sum(dim=-1)
assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
    atol=1e-5,
)
```

测试时应关闭 Dropout或调用 `model.eval()`，否则权重经过 Dropout 后行和不一定仍为 1。

## 10. MLP 与 Encoder Block

MLP：

```python
class MLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
```

Pre-Norm Encoder Block：

```python
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=dropout,
            projection_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### 10.1 用多个 Encoder Block 组成完整 Encoder

一个 Encoder Block 只完成一次 Attention 和 MLP 处理。完整 ViT Encoder 要把结构相同、但参数彼此独立的 Block 串联起来：

```text
x
→ Block 0
→ Block 1
→ ...
→ Block depth-1
→ 最终 LayerNorm
→ Encoder 输出
```

与本项目当前的 `Encoder1Block` 接口对应，可以写成：

```python
import torch
from torch import nn

from src.encoder import Encoder1Block


class Encoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()

        if depth <= 0:
            raise ValueError("depth must be a positive integer")

        self.blocks = nn.ModuleList([
            Encoder1Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(depth)
        ])

        # Pre-Norm Transformer 堆叠完成后通常再做一次最终归一化。
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 始终保持 (B, N, D)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x
```

`range(depth)` 依次产生：

```python
0, 1, 2, ..., depth - 1
```

这里不需要使用这些编号，所以按 Python 惯例将循环变量写成 `_`：

```python
for _ in range(depth)
```

列表推导式的含义可以展开成普通循环：

```python
blocks = []

for _ in range(depth):
    block = Encoder1Block(...)
    blocks.append(block)

self.blocks = nn.ModuleList(blocks)
```

关键点是 `Encoder1Block(...)` 位于循环内部。每轮循环都会调用一次构造函数，创建一个新的 Python 对象以及一套新的 `LayerNorm`、Attention、MLP 参数。因此，虽然所有 Block 的结构和超参数相同，但可训练参数彼此独立。

如果 `depth=3`，PyTorch 会把参数注册成类似下面的层次：

```text
blocks.0.norm1.weight
blocks.0.attention.qkv.weight
blocks.0.mlp.fc1.weight

blocks.1.norm1.weight
blocks.1.attention.qkv.weight
blocks.1.mlp.fc1.weight

blocks.2.norm1.weight
blocks.2.attention.qkv.weight
blocks.2.mlp.fc1.weight

norm.weight
norm.bias
```

这些不同的名字会出现在 `model.named_parameters()` 和 `model.state_dict()` 中，优化器也能找到并分别更新它们。

#### 会意外共享参数的错误写法

下面的代码只创建了一个 Block，然后把同一个对象引用放入列表多次：

```python
shared_block = Encoder1Block(...)
self.blocks = nn.ModuleList([shared_block] * depth)  # 错误：共享参数
```

下面这种写法也一样会共享：

```python
shared_block = Encoder1Block(...)
self.blocks = nn.ModuleList([
    shared_block
    for _ in range(depth)
])  # 错误：仍然是同一个对象
```

它们在前向传播时确实会执行 `depth` 次，但每次使用的都是同一套权重。这属于循环使用同一个模型，而不是堆叠多个独立层。

正确写法是在每次循环中重新执行构造函数：

```python
self.blocks = nn.ModuleList([
    Encoder1Block(...)
    for _ in range(depth)
])
```

`nn.ModuleList` 本身不负责前向传播，它主要负责向 PyTorch 注册其中的所有子模块。因此仍然要在 `forward` 中显式循环：

```python
for block in self.blocks:
    x = block(x)
```

每轮的输出都会赋值回 `x`，再作为下一层的输入。若有三个 Block，则等价于：

```python
x = self.blocks[0](x)
x = self.blocks[1](x)
x = self.blocks[2](x)
```

也可以使用 `nn.Sequential` 自动依次调用各层，但 `ModuleList + for` 更直观，也方便以后加入返回中间特征、梯度检查点或逐层调试等逻辑。

#### Encoder 的基础 pytest 测试

```python
import torch

from src.encoder import Encoder


def test_encoder_output_shape():
    model = Encoder(
        depth=4,
        embed_dim=64,
        num_heads=8,
        mlp_dim=256,
    )

    x = torch.randn(2, 65, 64)
    output = model(x)

    assert output.shape == (2, 65, 64)
    assert len(model.blocks) == 4


def test_encoder_blocks_do_not_share_parameters():
    model = Encoder(
        depth=2,
        embed_dim=64,
        num_heads=8,
        mlp_dim=256,
    )

    # 两个列表元素必须是不同的 Encoder1Block 对象。
    assert model.blocks[0] is not model.blocks[1]

    weight0 = model.blocks[0].attention.qkv.weight
    weight1 = model.blocks[1].attention.qkv.weight

    # 两个 Parameter 对象不同，底层存储地址也不同。
    assert weight0 is not weight1
    assert weight0.data_ptr() != weight1.data_ptr()
```

这里不要通过下面的方式判断是否共享：

```python
assert not torch.equal(weight0, weight1)
```

“数值是否相等”和“是否为同一个参数”是两件事。判断参数共享应检查对象身份或存储地址，即 `is` 和 `data_ptr()`。

完整 Encoder 不改变张量形状：

```text
输入： (B, N, D)
每层： (B, N, D) → (B, N, D)
输出： (B, N, D)
```

Block 的输出形状相同，是因为残差连接要求相加的两个张量形状完全一致；不同层通过各自独立的权重对特征进行逐层变换。

形状主线：

```text
(B,N,D)
→ Attention 分支 → (B,N,D) → 残差
→ MLP：D→M→D → (B,N,D) → 残差
→ (B,N,D)
```

### 10.2 Transformer：加入 CLS、位置编码并调用 Encoder

按本项目当前的文件划分，`transformer.py` 可以负责：

```text
图片
→ Patch Embedding
→ 添加 CLS token
→ 添加位置编码
→ Embedding Dropout
→ Encoder
→ 返回全部编码 token
```

先约定符号：

```text
B：batch size
C：输入通道数
H、W：图片高度和宽度
P：patch size
N：(H/P) × (W/P)，图片产生的 patch token 数量
D：embed_dim，每个 token 的特征维度
L：token 序列长度；加入 CLS 后 L=N+1
```

与当前 `PatchEmbedding` 和 `Encoder` 接口对应的示例：

```python
import torch
from torch import nn

from src.patch_embedding import PatchEmbedding
from src.encoder import Encoder


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.patch_embedding.num_patches + 1,
                embed_dim,
            )
        )

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.encoder = Encoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            mlp_dropout=mlp_dropout,
        )

        # 常见 ViT 初始化。第一版使用全零初始化也能运行，
        # 但截断正态分布更接近常见实现。
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W)
        x = self.patch_embedding(images)
        # x: (B, N, D)

        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # (1, 1, D) -> (B, 1, D)

        x = torch.cat((cls_tokens, x), dim=1)
        # (B, 1, D) 与 (B, N, D) 沿 token 维拼接
        # x: (B, N+1, D)

        if x.shape[1] != self.pos_embedding.shape[1]:
            raise ValueError(
                "Input token count does not match position embedding"
            )

        x = x + self.pos_embedding
        # (B, N+1, D) + (1, N+1, D)
        # 位置编码在 batch 维广播，结果仍为 (B, N+1, D)

        x = self.embedding_dropout(x)
        x = self.encoder(x)
        # Encoder 不改变形状：仍为 (B, N+1, D)

        return x
```

这个版本返回的是**全部 token**，不是分类结果：

```text
Transformer 输出：(B, N+1, D)
```

以 CIFAR-10 配置为例：

```text
images.shape = (2, 3, 32, 32)
patch_size = 4
embed_dim = 64

每边 patch 数 = 32/4 = 8
N = 8×8 = 64
加入一个 CLS token 后，L = N+1 = 65

最终全部 token：tokens.shape = (2, 65, 64)
```

#### 输出中的每一部分是什么

```python
tokens = model(images)       # (B, N+1, D)

cls_features = tokens[:, 0]  # (B, D)
patch_features = tokens[:, 1:]  # (B, N, D)
```

token 维的下标含义是：

```text
tokens[:, 0]    ：经过所有 Encoder 层处理后的 CLS token
tokens[:, 1]    ：第 1 个 patch token
tokens[:, 2]    ：第 2 个 patch token
...
tokens[:, N]    ：第 N 个 patch token
```

注意下面两个切片的形状不同：

```python
tokens[:, 0].shape      # (B, D)，token 维被索引掉
tokens[:, 0:1].shape    # (B, 1, D)，保留 token 维
```

做图像分类时，分类头 `nn.Linear` 通常需要二维输入，所以使用：

```python
cls_features = tokens[:, 0]   # (B, D)
logits = self.head(cls_features)
# logits: (B, num_classes)
```

对于 CIFAR-10：

```text
全部 token：  (B, 65, D)
取 CLS 后：   (B, D)
分类头输出：  (B, 10)
```

因此，“最终输出形状”取决于所说的是哪一层：

| 输出位置 | 形状 | 含义 |
|---|---:|---|
| PatchEmbedding 后 | `(B, N, D)` | 只有图像 patch token |
| 加 CLS 和位置编码后 | `(B, N+1, D)` | 完整输入 token 序列 |
| Encoder 后 | `(B, N+1, D)` | 全部 token 的编码结果 |
| `tokens[:, 0]` 后 | `(B, D)` | 每张图片的 CLS 特征 |
| 分类头后 | `(B, num_classes)` | 每张图片对各类别的 logits |

`CrossEntropyLoss` 需要送入分类 logits，而不是全部 token：

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

# logits: (B, num_classes)
# labels: (B,)，每个元素是类别编号
```

#### 为什么 Encoder 要保留全部 token

虽然最终分类只取 CLS token，但不能在进入 Encoder 前就扔掉 patch token。每层自注意力都会让 CLS token 查询并汇总所有 patch token 的信息：

```text
CLS ↔ patch 1
CLS ↔ patch 2
...
CLS ↔ patch N
```

经过多层交互后，最后的 `tokens[:, 0]` 才成为整张图片的特征表示。只有完成全部 Encoder 层之后，才取出 CLS token 用于分类。

#### Transformer 的基础 pytest 测试

```python
import torch

from src.transformer import VisionTransformer


def test_transformer_output_shape():
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
```

测试时可以额外验证不同 batch size。batch size 改变时，只有第 0 维改变：

```python
def test_transformer_different_batch_size():
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_layers=2,
        embed_dim=64,
        num_heads=8,
        mlp_dim=128,
    )

    images = torch.randn(7, 3, 32, 32)
    tokens = model(images)

    assert tokens.shape == (7, 65, 64)
```

如果决定让 `VisionTransformer` 直接代表完整分类模型，也可以在构造函数中增加：

```python
self.head = nn.Linear(embed_dim, num_classes)
```

并把 `forward` 的最后两行改为：

```python
cls_features = x[:, 0]
logits = self.head(cls_features)
return logits
```

此时模型对外输出的是 `(B, num_classes)`，而不是 `(B, N+1, D)`。两种设计都能工作，但测试和训练代码必须与所选接口保持一致。本项目在当前阶段建议先返回全部 token，确认 token 流程无误；下一步加入分类头时再返回 logits。

## 11. 完整 ViT 数据流

```text
images
(B,3,32,32)
    ↓ PatchEmbedding，patch=4，embed_dim=D
(B,64,D)
    ↓ 加 CLS token
(B,65,D)
    ↓ 加位置编码 (1,65,D)
(B,65,D)
    ↓ embedding dropout
(B,65,D)
    ↓ EncoderBlock × depth
(B,65,D)
    ↓ 最终 LayerNorm
(B,65,D)
    ↓ x[:,0]
(B,D)
    ↓ Linear(D,10)
(B,10)
```

CLS 和位置编码：

```python
b = x.shape[0]
cls = self.cls_token.expand(b, -1, -1)
x = torch.cat((cls, x), dim=1)
x = x + self.pos_embedding
```

取分类 token：

```python
x = x[:, 0]
logits = self.head(x)
```

## 12. CIFAR-10 数据管道

### 12.1 这一阶段要完成什么

模型训练前的数据链路是：

```text
CIFAR-10 磁盘文件
→ Dataset 按索引读取一张图片和一个标签
→ transform 做增强、转 Tensor、归一化
→ DataLoader 把多个样本组成 batch
→ 得到 images: (B,3,32,32) 和 labels: (B,)
→ 移动到与模型相同的 device
→ model(images) 得到 logits: (B,10)
```

这一阶段的验收目标：

1. 能自动下载并读取 CIFAR-10；
2. 训练、验证、测试三个集合职责明确；
3. 训练集使用随机增强，验证和测试集不使用随机增强；
4. DataLoader 能输出正确形状和数据类型；
5. 一个 batch 能送入完整模型并得到 `(B,10)` logits；
6. 数据目录不会被提交到 Git。

所需依赖可记录在 `requirements.txt`：

```text
torch
torchvision
pytest
```

安装：

```powershell
python -m pip install -r requirements.txt
```

常用导入：

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
```

### 12.2 Dataset、transform 和 DataLoader 的分工

三者不要混为一谈：

| 对象 | 作用 | 本项目中的例子 |
|---|---|---|
| `Dataset` | 定义“第 i 个样本怎样读取” | `datasets.CIFAR10` |
| `transform` | 处理一张刚读取的图片 | 裁剪、翻转、转 Tensor、归一化 |
| `DataLoader` | 取样、打乱、组成 batch、并行加载 | 输出 `(images, labels)` |

直接访问 Dataset：

```python
image, label = dataset[0]
```

在使用 `ToTensor()` 后，单个样本通常为：

```text
image.shape = (3,32,32)
image.dtype = torch.float32
label        = Python int，范围 0～9
```

DataLoader 会使用默认的整理逻辑把多个样本堆叠起来：

```text
B 个 (3,32,32)       → images: (B,3,32,32)
B 个整数类别编号      → labels: (B,)，dtype=torch.int64
```

### 12.3 CIFAR-10 的基本信息

```text
训练部分：50,000 张图片
测试部分：10,000 张图片
图片大小：32×32
通道数：3（RGB）
类别数：10
```

类别编号和名称：

```python
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
```

训练过程中不要反复查看测试集来选择模型。更规范的划分是：

```text
原训练部分 50,000
├── train 45,000：更新参数
└── val    5,000：选择超参数、观察过拟合

原测试部分 10,000
└── test  10,000：模型确定后做最终评估
```

### 12.4 图像变换与形状变化

第一版推荐使用经典、容易理解的增强：

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])
```

执行顺序就是 `Compose` 中从上到下的顺序：

```text
训练图片 PIL Image，32×32×3
→ RandomCrop：先 padding，再随机裁回 32×32
→ RandomHorizontalFlip：按概率水平翻转
→ ToTensor：(H,W,C)、uint8、[0,255]
             变为 (C,H,W)、float32、[0,1]
→ Normalize：逐通道执行 (x-mean)/std
→ (3,32,32)、float32
```

`Normalize` 不是把数值严格变成 `[-1,1]`，而是使每个通道大致以 0 为中心。公式是：

```python
normalized[channel] = (
    image[channel] - mean[channel]
) / std[channel]
```

验证集和测试集不能使用 `RandomCrop`、`RandomHorizontalFlip` 等随机增强，否则同一张图片每次验证可能不同，指标会发生无意义的波动。

### 12.5 创建可复现的 train、val、test 划分

一个容易忽略的问题是：`Subset` 只保存原 Dataset 和索引。如果 train、val 都引用同一个带随机增强的 Dataset，那么 val 也会被随机增强。

因此，为同一份 CIFAR-10 训练数据创建两个 Dataset 对象：

- `train_full` 使用 `train_transform`；
- `val_full` 使用 `eval_transform`；
- 两者使用互不重叠的索引。

完整函数可以放在 `src/data.py`：

```python
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
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, eval_transform


def build_cifar10_datasets(
    data_dir: str | Path = "data",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    train_transform, eval_transform = make_transforms()

    # 两个对象读取的是同一份磁盘数据，但使用不同 transform。
    train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    num_examples = len(train_full)
    num_val = int(num_examples * val_ratio)

    # 使用局部 Generator 固定索引划分，不依赖全局随机状态。
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        num_examples,
        generator=generator,
    ).tolist()

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    return train_dataset, val_dataset, test_dataset
```

这里的 `seed` 保证每次运行得到相同的 train/val 索引。它不会让训练增强永远相同；随机裁剪和随机翻转在不同 epoch 仍可产生不同结果。

检查划分：

```python
train_set, val_set, test_set = build_cifar10_datasets()

assert len(train_set) == 45_000
assert len(val_set) == 5_000
assert len(test_set) == 10_000

assert set(train_set.indices).isdisjoint(val_set.indices)
```

如果现阶段只想先跑通代码，也可以暂时直接使用完整训练集和测试集；但正式比较实验时建议保留验证集。

### 12.6 创建 DataLoader

继续在 `src/data.py` 中加入：

```python
def build_cifar10_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    train_set, val_set, test_set = build_cifar10_datasets(
        data_dir=data_dir,
        val_ratio=val_ratio,
        seed=seed,
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader
```

参数含义：

| 参数 | 含义 | 第一版建议 |
|---|---|---|
| `batch_size` | 每个训练 batch 的样本数 | 显存允许时可从 64 或 128 开始 |
| `shuffle` | 每个 epoch 重新打乱索引 | train 为 `True`，val/test 为 `False` |
| `num_workers` | 使用多少个子进程读取数据 | Windows 初次调试用 `0` |
| `pin_memory` | 使用页锁定 CPU 内存，帮助传往 CUDA | 有 CUDA 时开启 |
| `drop_last` | 丢掉最后一个不足 batch 的小批次 | 第一版设为 `False`，保留所有样本 |
| `persistent_workers` | epoch 间保留 worker | 仅在 `num_workers>0` 时开启 |

训练集需要 `shuffle=True`，因为模型不应长期按照固定样本顺序更新。val/test 必须 `shuffle=False`，便于结果复查，而且它们不更新参数。

`drop_last=False` 时，最后一个 batch 可能比其他 batch 小。例如 45,000 个样本、batch size 为 128：

```text
前面的 batch：images.shape = (128,3,32,32)
最后的 batch：images.shape 可能小于 (128,3,32,32)
```

模型代码必须从 `x.shape[0]` 动态取得 `B`，不能把 batch size 写死。

### 12.7 第一次运行和 batch 检查

在项目根目录运行一个临时检查，或写入 `tests/test_data.py`：

```python
import torch

from src.data import build_cifar10_loaders


train_loader, val_loader, test_loader = build_cifar10_loaders(
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
```

第一次运行时 `download=True` 会从网络下载数据。已经存在完整数据后，torchvision 会复用它，而不是每次重新下载。

注意：这类测试依赖网络或本地数据，不适合作为每次都必须运行的纯单元测试。可以把下载单独执行一次，之后再运行数据形状测试。

### 12.8 把一个 batch 送进模型

数据形状正确后，做一次端到端检查：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
model.eval()

images, labels = next(iter(train_loader))
images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

with torch.no_grad():
    logits = model(images)

assert logits.shape == (images.shape[0], 10)
assert labels.shape == (images.shape[0],)
assert torch.isfinite(logits).all()
```

如果当前 `VisionTransformer` 返回的是全部 token：

```text
(B,65,D)
```

那么它还不能直接送进 `CrossEntropyLoss`。必须先完成分类路径：

```python
tokens = backbone(images)       # (B,65,D)
cls_features = tokens[:, 0]     # (B,D)
logits = head(cls_features)     # (B,10)
```

训练代码最终应该接收 `(B,10)`，标签保持 `(B,)`，不要对标签做 one-hot：

```python
loss = torch.nn.functional.cross_entropy(logits, labels)
```

### 12.9 Windows 上使用 DataLoader

第一次调试使用：

```python
num_workers = 0
```

确认无误后可以尝试 `2` 或 `4`，并比较实际加载速度。`num_workers` 并不是越大越快。

当 `num_workers > 0` 时，创建和遍历 DataLoader 的入口应放在：

```python
def main():
    train_loader, val_loader, test_loader = build_cifar10_loaders(
        num_workers=4,
    )

    for images, labels in train_loader:
        pass


if __name__ == "__main__":
    main()
```

否则 Windows 的多进程启动方式可能反复执行主文件或报启动进程错误。

### 12.10 常见错误

#### 输入形状变成 `(B,32,32,3)`

PyTorch 卷积需要 NCHW：

```text
正确：(B,3,32,32)
错误：(B,32,32,3)
```

使用 torchvision 的 `ToTensor()` 会把 HWC 转为 CHW。

#### 图像仍是 `uint8`

检查是否遗漏：

```python
transforms.ToTensor()
```

#### 在 `ToTensor` 前调用 `Normalize`

经典 transforms 管道中 `Normalize` 接收 Tensor，因此应放在 `ToTensor()` 后面。

#### 对验证集也使用随机增强

这会让验证输入不断变化。train 与 val 应创建两个 Dataset 对象并分别使用不同 transform。

#### 使用测试集反复挑选超参数

这相当于把测试集信息泄漏进模型选择。训练期间看 val，最终确定模型后再看 test。

#### 把整个数据集提前移动到 GPU

Dataset 和 DataLoader 通常保留在 CPU；训练循环中只把当前 batch 移到 GPU。

#### 将数据集提交到 Git

项目 `.gitignore` 应包含：

```text
data/
datasets/
```

当前项目已经忽略这两个目录。

### 12.11 数据管道完成标准

完成后逐项确认：

- `len(train_set) == 45000`、`len(val_set) == 5000`、`len(test_set) == 10000`；
- train 和 val 索引没有交集；
- train 使用随机增强，val/test 使用确定性预处理；
- `images` 为 `(B,3,32,32)`、`float32`；
- `labels` 为 `(B,)`、`int64`，取值范围 0～9；
- 一个 batch 不包含 `NaN` 或无穷值；
- 完整模型对一个 batch 输出 `(B,10)`；
- Windows 下 `num_workers=0` 能稳定运行；
- `data/` 未进入 Git 暂存区。

## 13. 设备管理

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
```

训练中：

```python
images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

模型参数和输入必须位于同一个设备。

检查：

```python
print(device)
print(next(model.parameters()).device)
print(images.device)
```

## 14. 随机种子与可复现性

```python
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

需要更强的确定性时，可以进一步设置：

```python
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

注意：严格确定性可能降低速度，且不同硬件、PyTorch/CUDA 版本之间不一定逐位一致。

## 15. 一次训练 step

```python
model.train()

optimizer.zero_grad(set_to_none=True)

logits = model(images)                    # (B,10)
loss = F.cross_entropy(logits, labels)    # 标量

loss.backward()
optimizer.step()
```

正确顺序：

```text
清空旧梯度
→ 前向传播
→ 计算 loss
→ backward 写入 parameter.grad
→ optimizer.step 更新参数
```

PyTorch 默认累加 `.grad`，因此常规训练每一步都必须清空旧梯度。

## 16. 完整训练循环骨架

```python
for epoch in range(num_epochs):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=-1) == labels
        ).sum().item()
        total_count += batch_size

    train_loss = total_loss / total_count
    train_accuracy = total_correct / total_count
```

使用 `loss.item() * batch_size` 累计样本 loss，最后除以总样本数，能正确处理不同大小的 batch。

## 17. 验证循环

```python
model.eval()

total_loss = 0.0
total_correct = 0
total_count = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=-1) == labels
        ).sum().item()
        total_count += batch_size

test_loss = total_loss / total_count
test_accuracy = total_correct / total_count
```

必须同时理解：

```text
model.eval()
    关闭 Dropout 等训练行为

torch.no_grad()
    不记录反向图，节省内存和计算
```

## 18. 优化器、梯度裁剪和学习率调度

AdamW 示例：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.05,
)
```

梯度裁剪必须在 `backward` 后、`step` 前：

```python
loss.backward()

grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

optimizer.step()
```

简单余弦调度：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
)
```

若按 epoch 调度：

```python
for epoch in range(num_epochs):
    train_one_epoch(...)
    evaluate(...)
    scheduler.step()
```

读取当前学习率：

```python
current_lr = optimizer.param_groups[0]["lr"]
```

第一版可以先使用固定学习率，训练流程正确后再加入 warmup 和 cosine decay。

## 19. Checkpoint 保存与恢复

保存完整训练状态：

```python
torch.save(
    {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_accuracy": best_accuracy,
    },
    checkpoint_path,
)
```

恢复：

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])

start_epoch = checkpoint["epoch"] + 1
step = checkpoint["step"]
best_accuracy = checkpoint["best_accuracy"]
```

只做推理至少需要：

```text
模型配置 + model.state_dict()
```

继续训练还需要优化器、调度器和进度状态。

## 20. 常用检查与调试

### 20.1 检查参数量

```python
num_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

num_trainable = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
```

### 20.2 检查梯度

```python
for name, parameter in model.named_parameters():
    if parameter.grad is None:
        print("no gradient:", name)
```

梯度范数：

```python
grad_norm_sq = 0.0

for parameter in model.parameters():
    if parameter.grad is not None:
        grad_norm_sq += parameter.grad.norm(2).item() ** 2

grad_norm = grad_norm_sq ** 0.5
```

### 20.3 检查有限数值

```python
assert torch.isfinite(logits).all()
assert torch.isfinite(loss)
```

### 20.4 检查参数确实更新

```python
old_weight = model.head.weight.detach().clone()

loss.backward()
optimizer.step()

difference = (
    model.head.weight.detach() - old_weight
).abs().max()

assert difference > 0
```

### 20.5 检查训练/推理模式

```python
model.train()
assert model.training

model.eval()
assert not model.training
```

### 20.6 小数据过拟合测试

```python
small_indices = list(range(32))
small_set = Subset(train_set, small_indices)

small_loader = DataLoader(
    small_set,
    batch_size=32,
    shuffle=True,
)
```

关闭强随机增强，反复训练这 32 张图片。若训练准确率长期无法接近 100%，优先排查实现、学习率、标签和参数更新。

## 21. PyTest 基础

测试文件中的函数以 `test_` 开头：

```python
import torch

from src.patch_embedding import PatchEmbedding


def test_patch_embedding_shape():
    module = PatchEmbedding(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
    )

    images = torch.randn(2, 3, 32, 32)
    tokens = module(images)

    assert tokens.shape == (2, 64, 64)
```

运行：

```text
pytest -q
```

建议每个模块至少测试：

- 正确输入的输出形状；
- 非法配置是否报错；
- 输出是否有限；
- 反向传播后关键参数是否有梯度。

## 22. 推荐的两套模型配置

### 22.1 最小调试配置

```text
image_size = 32
patch_size = 4
num_classes = 10
embed_dim = 64
depth = 2
num_heads = 4
mlp_dim = 128
dropout = 0.1
attention_dropout = 0.0
```

用途：

- 随机张量形状测试；
- 一次反向传播；
- 小数据过拟合；
- CPU 上快速调试。

### 22.2 CIFAR-10 正式实验起点

```text
image_size = 32
patch_size = 4
num_classes = 10
embed_dim = 192
depth = 6
num_heads = 3
mlp_dim = 768
dropout = 0.1
attention_dropout = 0.0
```

具体配置需要根据 GPU 显存和训练时间调整。先证明小模型训练正确，再扩大模型。

## 23. 完整复现顺序与验收标准

### 阶段 1：Patch Embedding

文件：`src/patch_embedding.py`

实现：

```text
Conv2d(kernel=P, stride=P)
→ flatten 空间维
→ transpose 为 (B,N,D)
```

验收：

```text
(2,3,32,32) → (2,64,64)  # P=4,D=64
```

### 阶段 2：多头自注意力

文件：`src/attention.py`

实现：

```text
QKV 投影
→ 拆分多头
→ 缩放点积
→ Softmax
→ 汇总 V
→ 合并多头
→ 输出投影
```

验收：

```text
(2,65,64) → (2,65,64)
Attention Softmax 行和约等于 1
qkv.weight 能获得梯度
```

### 阶段 3：MLP 和 Encoder Block

文件：`src/transformer.py`

实现：

```text
x + Attention(LayerNorm(x))
x + MLP(LayerNorm(x))
```

验收：

```text
(2,65,64) → (2,65,64)
两个残差分支形状一致
```

### 阶段 4：完整 ViT

文件：`src/vit.py`

实现：

```text
PatchEmbedding
→ CLS
→ Position Embedding
→ Encoder
→ final LayerNorm
→ CLS feature
→ classification head
```

验收：

```text
(2,3,32,32) → (2,10)
参数量合理
所有输出有限
```

### 阶段 5：随机 batch 反向传播

验收：

```text
loss 为标量且有限
Patch Embedding、Attention、MLP、head 都有梯度
optimizer.step 后参数发生变化
```

### 阶段 6：少量 CIFAR-10 过拟合

验收：

```text
16～32 张图片的训练准确率能够接近 100%
```

### 阶段 7：完整训练与验证

文件：`train.py`、`evaluate.py`

验收：

```text
训练 loss 总体下降
验证准确率明显高于 10% 随机水平
训练/验证模式切换正确
指标按总样本数加权汇总
```

### 阶段 8：Checkpoint 和日志

验收：

```text
模型能够保存并恢复
恢复后同一输入的推理结果一致
能够继续训练
记录 loss、accuracy、learning rate
```

### 阶段 9：可选工程功能

基础版本稳定后再考虑：

```text
warmup + cosine schedule
更强数据增强
混合精度
梯度累积
TensorBoard
torch.compile
多 GPU DDP
预训练权重转换
位置编码插值
```

## 24. 常见错误速查

### Conv2d 报通道错误

检查输入是不是：

```text
(B,C,H,W)
```

而不是 JAX 的 `(B,H,W,C)`。

### Linear 矩阵乘法形状错误

检查最后一维是否等于 `in_features`。

### 残差相加失败

检查两个分支是否都是：

```text
(B,N,D)
```

### Attention reshape 失败

检查：

```python
embed_dim % num_heads == 0
```

### Softmax 方向错误

应使用：

```python
scores.softmax(dim=-1)
```

### 验证结果随机变化

检查是否调用：

```python
model.eval()
```

### 显存持续增加

检查验证是否使用：

```python
with torch.no_grad():
```

以及是否把带计算图的 loss/tensor 长期保存到 Python 列表。

### 参数没有更新

检查：

```text
optimizer.zero_grad
loss.backward
optimizer.step
```

的顺序，以及关键参数的 `.grad` 是否为 `None`。

### DataLoader 在 Windows 报多进程错误

先设置：

```python
num_workers=0
```

并确保训练入口位于：

```python
if __name__ == "__main__":
    main()
```

## 25. 第一版暂不实现的官方功能

以下功能来自官方 JAX 工程，但不会进入第一版 PyTorch 复现：

- ResNet-ViT 混合骨架；
- `token`、`gap`、`unpooled` 等多种 classifier 模式；
- `pre_logits + tanh` 表示层；
- AugReg checkpoint 自动选择；
- 官方 JAX `.npz` 参数转换；
- 输入分辨率变化时的位置编码插值；
- 多主机、多设备 `pmap/pmean`；
- 云存储和 CLU 实验平台支持。

第一版固定：

```text
标准 ViT
CIFAR-10
CLS token 分类
单设备
从零训练
```

## 26. 常用官方文档

- PyTorch 文档：<https://docs.pytorch.org/docs/stable/>
- `nn.Module`：<https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html>
- `torch.Tensor`：<https://docs.pytorch.org/docs/stable/tensors.html>
- `torch.nn`：<https://docs.pytorch.org/docs/stable/nn.html>
- `torch.optim`：<https://docs.pytorch.org/docs/stable/optim.html>
- DataLoader：<https://docs.pytorch.org/docs/stable/data.html>
- torchvision CIFAR-10：<https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>
- torchvision transforms：<https://docs.pytorch.org/vision/stable/transforms.html>
- torchvision 数据集与变换：<https://docs.pytorch.org/vision/stable/index.html>
- 保存和加载模型：<https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html>
- PyTest：<https://docs.pytest.org/>

## 27. 当前下一步

模型结构和 CIFAR-10 数据读取已经基本完成，下一阶段按以下顺序推进：

1. 修正并验证 train/val 数据划分；
2. 编写“单个训练 step”测试，证明参数确实更新；
3. 编写 `src/engine.py`，封装一个 epoch 的训练和验证；
4. 用 16～32 个样本做过拟合实验；
5. 编写 `configs/cifar10.py`，集中保存配置；
6. 完善 `utils.py`，加入随机种子、参数统计和 checkpoint；
7. 完成 `train.py`，串联模型、数据、优化器、训练和验证；
8. 完成 `evaluate.py`，加载最佳 checkpoint 并只在 test 集评估；
9. 最后再考虑 warmup、混合精度、TensorBoard 等增强功能。

不要跳过单步更新和小数据过拟合，直接开始几十或几百个 epoch 的正式训练。前两项是发现实现错误成本最低的阶段。

推荐按下面的提交边界保存版本：

```text
提交 1：完成模型结构与形状测试
提交 2：完成 CIFAR-10 数据管道与 batch 检查
提交 3：完成单步反向传播与少量样本过拟合
提交 4：完成正式训练、验证、日志和 checkpoint
```

## 28. 后续文件及其职责

推荐把后续代码组织为：

```text
vit-pytorch-from-scratch/
├── configs/
│   └── cifar10.py              # 模型、数据和训练超参数
├── src/
│   ├── data.py                 # 已完成：Dataset 与 DataLoader
│   ├── engine.py               # train_one_epoch、evaluate_one_epoch
│   └── ...                     # 已完成的模型模块
├── tests/
│   ├── test_train_step.py      # 梯度、参数更新、loss 有限
│   └── ...                     # 已完成的形状测试
├── train.py                    # 正式训练入口
├── evaluate.py                 # 最终测试入口
├── utils.py                    # seed、参数量、checkpoint
└── requirements.txt
```

调用链：

```text
python train.py
→ 读取 Config
→ seed_everything
→ build_cifar10_dataloaders
→ 创建 VisionTransformer(classifier="cls")
→ 创建 loss、optimizer、scheduler
→ 循环 epoch
   ├── train_one_epoch
   ├── evaluate_one_epoch(val)
   ├── scheduler.step
   ├── 保存 last checkpoint
   └── 若 val accuracy 更高，保存 best checkpoint

python evaluate.py
→ 用相同 Config 创建模型
→ 加载 best checkpoint
→ evaluate_one_epoch(test)
→ 输出最终 test loss/accuracy
```

### 28.1 进入训练前先修正当前数据划分

若验证集比例为 `0.1`，应有 45,000 个训练样本、5,000 个验证样本。索引应写为：

```python
val_indices = indices[:val_size]
train_indices = indices[val_size:]
```

不要写反：

```python
train_indices = indices[:val_size]   # 这只得到 5,000 个
val_indices = indices[val_size:]     # 这会得到 45,000 个
```

加入测试：

```python
assert len(train_set) == 45_000
assert len(val_set) == 5_000
assert set(train_set.indices).isdisjoint(val_set.indices)
```

## 29. 配置文件 `configs/cifar10.py`

配置的作用是避免数字散落在多个文件中。第一版可使用标准库 `dataclass`：

```python
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
```

使用：

```python
from configs.cifar10 import get_config

config = get_config()
print(config.embed_dim)
print(config.learning_rate)
```

`frozen=True` 表示创建后不应随意修改字段，有助于避免训练中途误改配置。它不是必须的；如果需要在命令行覆盖配置，可以先移除它。

配置值只是合理起点，不保证是 CIFAR-10 上的最佳超参数。第一轮目标仍然是验证代码正确，而不是追求最高准确率。

## 30. 单步训练测试 `tests/test_train_step.py`

它回答四个问题：

1. 模型是否输出 `(B,10)`；
2. loss 是否为有限标量；
3. 分类头是否获得梯度；
4. `optimizer.step()` 后参数是否改变。

```python
import torch
import torch.nn.functional as F

from src.transformer import VisionTransformer


def test_one_train_step_updates_parameters():
    torch.manual_seed(42)

    # 测试使用小模型，避免单元测试太慢。
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
```

这里使用 `classifier="cls"` 非常重要。若模型返回 `(B,65,D)` tokens，就不能直接与 `(B,)` 标签计算普通图像分类交叉熵。

运行：

```powershell
python -m pytest tests/test_train_step.py -v
```

## 31. 训练和验证引擎 `src/engine.py`

### 31.1 为什么单独建立 engine

`train.py` 负责“统筹实验”，`engine.py` 负责“怎样训练或验证一个 epoch”。这样 `evaluate.py` 可以直接复用验证函数，而不复制代码。

### 31.2 训练一个 epoch

```python
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
```

不要把每个 batch 的平均 loss 再简单平均，因为最后一个 batch 可能较小。这里把 batch mean loss 乘回 `batch_size`，最后再除以总样本数。

### 31.3 验证一个 epoch

```python
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
```

验证阶段的三项关键区别：

```text
model.eval()             关闭 Dropout 等训练行为
torch.inference_mode()   不构建梯度图
没有 optimizer.step()    不更新任何参数
```

`inference_mode` 比 `no_grad` 限制更强，适合纯验证和推理；第一版使用二者之一都可以。

## 32. 工具函数 `utils.py`

### 32.1 随机种子

```python
import random
from pathlib import Path

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### 32.2 参数量统计

```python
def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
```

### 32.3 保存 checkpoint

```python
def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_accuracy: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "best_val_accuracy": best_val_accuracy,
        },
        path,
    )
```

### 32.4 恢复 checkpoint

```python
def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> tuple[int, float]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_accuracy = checkpoint["best_val_accuracy"]
    return start_epoch, best_val_accuracy
```

只加载可信来源的 checkpoint。`map_location=device` 用于把张量映射到当前设备；`weights_only=True` 限制可反序列化的对象类型。当前保存的字典只包含张量、基本类型和各类 `state_dict`，适合这种方式。

两种 checkpoint 的职责：

```text
last.pt：每个 epoch 覆盖，程序中断后继续训练
best.pt：仅当 val accuracy 提升时覆盖，最终测试使用
```

## 33. 完整训练入口 `train.py`

下面的代码展示各模块怎样串起来。实现时可以先不加 resume 和复杂日志。

```python
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
        classifier="cls",
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
```

运行：

```powershell
python train.py
```

`CrossEntropyLoss` 接收未经 softmax 的 logits；不要在模型分类头后手动调用 softmax。其典型形状是：

```text
logits: (B,10)，float
labels: (B,)，long，类别编号 0～9
loss:   ()，标量
```

当前示例按 epoch 调用一次 `scheduler.step()`，因此 `T_max=config.num_epochs`。如果以后改成每个 batch 调用 scheduler，就必须相应地用总 step 数配置调度周期，不能混用。

## 34. 最终测试入口 `evaluate.py`

测试集只在模型和超参数确定后使用：

```python
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
```

注意：如果训练时使用了非零 `label_smoothing`，验证 loss 是否也使用相同设置要提前统一约定。准确率计算不受这个选项影响。第一版可以全部设为 `0.0`。

## 35. 正式训练前的小数据过拟合

### 35.1 为什么要做

小数据过拟合不是为了获得泛化能力，而是验证整条链路是否具备学习能力。如果 16～32 张固定图片都无法记住，长时间训练通常只会浪费算力。

### 35.2 建议设置

```text
样本数：16～32
数据增强：关闭随机裁剪和随机翻转
Dropout：全部设为 0
weight_decay：0
模型：可缩小到 depth=2、D=64、heads=4、mlp_dim=128
目标：训练准确率逐渐接近 100%，loss 明显下降
```

固定小数据集：

```python
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.data import make_transforms


_, eval_transform = make_transforms()

dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=eval_transform,
)

small_set = Subset(dataset, list(range(32)))
small_loader = DataLoader(
    small_set,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
```

反复对同一个 `small_loader` 调用 `train_one_epoch`。若失败，按顺序检查：

1. 模型是否为 `classifier="cls"`；
2. logits 和 labels 形状是否为 `(B,10)`、`(B,)`；
3. 是否调用 `model.train()`；
4. 是否执行 `zero_grad → forward → loss → backward → step`；
5. 关键参数的 `.grad` 是否为 `None` 或非有限值；
6. 学习率是否过小或过大；
7. 数据和标签是否对应；
8. 是否错误地在训练函数上使用了 `no_grad`/`inference_mode`。

## 36. 第一版训练完成标准

依次确认：

- 数据划分是 train=45,000、val=5,000、test=10,000；
- 所有已有 pytest 通过；
- 单个训练 step 能产生有限 loss 和梯度，并改变参数；
- 16～32 个固定样本可以明显过拟合；
- `train_one_epoch` 和 `evaluate_one_epoch` 正确处理最后的小 batch；
- 日志同时显示 epoch、lr、train loss/accuracy、val loss/accuracy；
- `last.pt` 可以恢复训练状态；
- `best.pt` 对应最高 val accuracy；
- 正式训练期间不使用 test 指标选择模型；
- `evaluate.py` 能加载 `best.pt` 并输出最终 test 指标；
- checkpoint、data 和日志目录没有被 Git 提交。

完成这些内容后，第一版“从零实现并训练 ViT”才形成闭环。之后再逐项加入 warmup、自动混合精度、TensorBoard、命令行参数和更强数据增强，每次只加一个功能并重新验证。
