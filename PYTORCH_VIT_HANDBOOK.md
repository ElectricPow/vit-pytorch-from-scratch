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
│   ├── transformer.py          # MLP、EncoderBlock、Encoder
│   └── vit.py                  # CLS、位置编码、完整 ViT、分类头
├── tests/
│   ├── test_patch_embedding.py
│   ├── test_attention.py
│   ├── test_transformer.py
│   └── test_vit.py
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

常用库：

```python
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
```

基础预处理示例：

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

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])
```

创建数据集：

```python
train_set = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform,
)

test_set = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)
```

创建 DataLoader：

```python
train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

test_loader = DataLoader(
    test_set,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)
```

Windows 初次调试时若 DataLoader 多进程报错，先使用：

```python
num_workers=0
```

训练标签是整数：

```text
labels.shape = (B,)
labels.dtype = torch.int64
```

PyTorch 的 `F.cross_entropy(logits, labels)` 直接接受整数类别，不需要 one-hot。

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
- torchvision 数据集与变换：<https://docs.pytorch.org/vision/stable/index.html>
- 保存和加载模型：<https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html>
- PyTest：<https://docs.pytest.org/>

## 27. 当前下一步

从 `src/patch_embedding.py` 开始：

1. 明确构造参数和输入输出形状；
2. 使用 `nn.Conv2d` 实现不重叠 Patch 投影；
3. 使用 `flatten(2)` 和 `transpose(1, 2)` 生成 `(B,N,D)`；
4. 编写 `tests/test_patch_embedding.py`；
5. 验证正确输入和非法输入；
6. 测试通过后再进入手写多头注意力。
