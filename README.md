# ViT PyTorch From Scratch

使用 PyTorch 从零实现 Vision Transformer（ViT），并搭建 CIFAR-10 图像分类所需的数据、训练、验证、测试和 checkpoint 流程。

本项目是本科阶段进入实验室后的源码学习与复现实践。实现过程中没有直接调用 `nn.MultiheadAttention` 或 `nn.TransformerEncoder` 隐藏核心结构，而是手工完成 Patch Embedding、多头自注意力、MLP、Encoder Block 和完整 ViT，重点理解每一步矩阵运算、张量形状以及 PyTorch 的参数管理方式。

> 当前状态：模型与训练代码闭环已经完成，各模块形状、真实 CIFAR-10 batch 前向传播和单步反向更新均已验证；尚未提交完整 100 epoch 的正式训练精度，因此本文不报告未经实验得到的准确率。

## 本周完成内容

- 从 JAX/Flax 官方 ViT 源码出发，梳理 ViT 的模块组成和数据流；
- 学习 PyTorch 的 `nn.Module`、`nn.Parameter`、`nn.ModuleList` 和自动求导；
- 使用 `Conv2d` 实现不重叠 Patch Embedding；
- 手工实现矩阵形式的多头自注意力，包括 QKV 投影、拆分多头、缩放点积和多头合并；
- 实现 GELU MLP、Pre-Norm Encoder Block 和多层 Encoder；
- 加入可训练的 CLS token、位置编码、最终 LayerNorm 和分类头；
- 支持返回全部编码 token 或 CIFAR-10 分类 logits；
- 完成 CIFAR-10 下载、增强、归一化和可复现的 train/val/test 划分；
- 完成训练与验证循环、AdamW、梯度裁剪、余弦学习率调度；
- 完成 `last.pt` / `best.pt` checkpoint 保存和最佳模型测试入口；
- 为核心模块编写形状测试，并验证一次反向传播能够真实更新模型参数；
- 整理了一份面向初学者的 [PyTorch ViT 复现手册](PYTORCH_VIT_HANDBOOK.md)。

## 模型结构

默认配置面向 `32×32` 的 CIFAR-10 图片：

```text
Image (B,3,32,32)
  │
  ├─ Conv2d(kernel=4, stride=4)
  ▼
Patch tokens (B,64,192)
  │
  ├─ prepend learnable CLS token
  ├─ add learnable position embedding
  ▼
Token sequence (B,65,192)
  │
  ├─ Encoder Block × 6
  │    ├─ LayerNorm
  │    ├─ Multi-Head Self-Attention
  │    ├─ Residual connection
  │    ├─ LayerNorm
  │    ├─ MLP: 192 → 768 → 192
  │    └─ Residual connection
  ├─ final LayerNorm
  ▼
Encoded tokens (B,65,192)
  │
  ├─ take tokens[:, 0]
  ▼
CLS feature (B,192)
  │
  ├─ Linear(192,10)
  ▼
Logits (B,10)
```

### 多头注意力形状

默认 `embed_dim=192`、`num_heads=3`，所以每个 head 的维度为 `64`：

```text
x                  (B,65,192)
qkv projection     (B,65,576)
split q, k, v      3 × (B,3,65,64)
q @ k.transpose    (B,3,65,65)
softmax(scores)    (B,3,65,65)
attention @ v      (B,3,65,64)
merge heads        (B,65,192)
output projection  (B,65,192)
```

注意力计算为：

```math
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}\right)V
```

### 输出模式

`VisionTransformer` 支持两种输出：

```python
# 返回全部 token：(B, N+1, D)
model = VisionTransformer(..., classifier="token")

# 返回分类 logits：(B, num_classes)
model = VisionTransformer(..., classifier="cls", num_classes=10)
```

训练 CIFAR-10 时必须使用 `classifier="cls"`。

## 默认配置

配置集中在 [`configs/cifar10.py`](configs/cifar10.py)：

| 项目 | 默认值 |
|---|---:|
| Image size | 32 |
| Patch size | 4 |
| Patch 数量 | 64 |
| Token 序列长度 | 65（包含 CLS） |
| Embedding dimension | 192 |
| Encoder depth | 6 |
| Attention heads | 3 |
| Head dimension | 64 |
| MLP dimension | 768 |
| Classes | 10 |
| Trainable parameters | 2,693,578 |
| Batch size | 128 |
| Epochs | 100 |
| Optimizer | AdamW |
| Initial learning rate | `3e-4` |
| Weight decay | `0.05` |
| Scheduler | CosineAnnealingLR |
| Minimum learning rate | `1e-6` |
| Gradient clipping | `1.0` |

这些参数是用于验证完整流程的初始配置，不代表已经完成系统调优。

## 数据处理

CIFAR-10 包含 50,000 张训练图片和 10,000 张测试图片。项目使用固定随机种子，将原训练部分划分为：

```text
train: 45,000
val:    5,000
test:  10,000
```

训练变换：

```text
RandomCrop(32, padding=4)
→ RandomHorizontalFlip
→ ToTensor
→ Normalize(CIFAR10_MEAN, CIFAR10_STD)
```

验证和测试不使用随机增强，只执行 `ToTensor` 和 `Normalize`。训练集与验证集使用不同的 Dataset 对象和互不重叠的索引，避免验证集意外继承训练随机增强。

DataLoader 输出：

```text
images: (B,3,32,32), torch.float32
labels: (B,),          torch.int64，取值 0～9
```

## 训练流程

一次训练 step 的顺序为：

```text
清空旧梯度
→ 前向传播得到 logits
→ CrossEntropyLoss
→ backward
→ 全局梯度范数裁剪
→ AdamW 更新参数
```

每个 epoch 后在验证集计算按样本数加权的 loss 和 accuracy，然后更新余弦学习率。训练过程中保存：

```text
checkpoints/last.pt   # 最近一个 epoch，用于恢复训练状态
checkpoints/best.pt   # 验证准确率最高的版本，用于最终测试
```

checkpoint 包含模型、优化器、scheduler、epoch 和最佳验证准确率状态。

## 项目结构

```text
vit-pytorch-from-scratch/
├── configs/
│   └── cifar10.py              # 模型、数据和训练配置
├── src/
│   ├── patch_embedding.py      # 图片切块与 Patch 投影
│   ├── attention.py            # 手写多头自注意力
│   ├── mlp.py                  # Transformer MLP
│   ├── encoder.py              # Encoder Block 与多层 Encoder
│   ├── transformer.py          # CLS、位置编码、Encoder 与分类头
│   ├── data.py                 # CIFAR-10 Dataset 和 DataLoader
│   └── engine.py               # 单个 epoch 的训练和验证
├── tests/
│   ├── test_patch_embedding.py
│   ├── test_attention.py
│   ├── test_mlp.py
│   ├── test_encoder.py
│   ├── test_transformer.py
│   ├── test_data.py
│   └── test_train_step.py
├── train.py                    # 正式训练入口
├── evaluate.py                 # 最佳 checkpoint 测试入口
├── utils.py                    # seed、参数统计、checkpoint
├── PYTORCH_VIT_HANDBOOK.md     # 学习与复现手册
├── requirements.txt
└── README.md
```

## 环境安装

项目使用了 Python 3.10 的类型标注语法，推荐 Python 3.10 或更高版本。

```powershell
git clone https://github.com/ElectricPow/vit-pytorch-from-scratch.git
cd vit-pytorch-from-scratch

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

如果需要 CUDA，请根据本机 CUDA 和显卡环境，从 PyTorch 官方安装页面选择匹配的安装命令，再安装其余依赖。

## 运行测试

在项目根目录执行：

```powershell
python -m pytest -v
```

测试覆盖：

- Patch Embedding 输出形状；
- 多头自注意力输入输出形状；
- MLP 和 Encoder Block 形状；
- 不同 Encoder Block 的参数不共享；
- 完整 token 序列及 CLS 切片形状；
- CIFAR-10 batch 的数据类型和分类输出；
- loss、梯度有限性和 optimizer 参数更新。

`test_data.py` 会访问 CIFAR-10，本地没有数据时会自动下载，因此第一次运行需要网络和磁盘空间。

不要直接执行：

```powershell
python tests/test_data.py
```

应从项目根目录使用 `python -m pytest`，确保 `src` 可以被正确导入。

## 开始训练

确认配置中的分类模式为：

```python
classifier = "cls"
```

然后执行：

```powershell
python train.py
```

程序会：

1. 固定随机种子；
2. 下载或读取 CIFAR-10；
3. 创建 train/val DataLoader；
4. 创建 ViT、交叉熵、AdamW 和余弦 scheduler；
5. 循环训练与验证；
6. 打印 epoch、学习率、loss 和 accuracy；
7. 保存最近和最佳 checkpoint。

日志格式示例：

```text
epoch 001/100 lr=0.0003 train_loss=... train_acc=... val_loss=... val_acc=...
```

## 最终测试

训练产生 `checkpoints/best.pt` 后执行：

```powershell
python evaluate.py
```

输出：

```text
test_loss=...
test_accuracy=...
```

测试集只用于模型与超参数确定后的最终评估，不用于反复选择训练配置。

## 当前验证结果

本次提交前已完成以下本地验证：

| 检查项 | 结果 |
|---|---|
| Patch Embedding 形状 | 通过 |
| Multi-Head Self-Attention 形状 | 通过 |
| MLP 形状 | 通过 |
| 单层与多层 Encoder 形状 | 通过 |
| Encoder Block 参数独立性 | 通过 |
| Transformer token 输出形状 | 通过 |
| 单步 loss、梯度和参数更新 | 通过 |
| CIFAR-10 45k/5k/10k 划分 | 通过 |
| 真实 batch `(128,3,32,32) → (128,10)` | 通过 |

完整长周期训练指标尚待实际运行。后续得到可复现结果后，应记录硬件、软件版本、随机种子、最佳 epoch、验证准确率和最终测试准确率。

## 学习过程总结

这次复现最重要的收获不是把模块名称照搬一遍，而是建立了从论文公式到工程代码的对应关系：

1. 理解图片怎样通过卷积变成 token 序列；
2. 将逐 token 的 QKV 公式转换为带 batch 和多头维度的矩阵运算；
3. 理解残差连接要求 Attention 和 MLP 保持 `(B,N,D)`；
4. 使用 `ModuleList` 创建参数独立的多层 Encoder；
5. 理解 CLS token 是一份共享可训练参数，但在 batch 中为每张图片形成独立计算图；
6. 区分全部 token `(B,N+1,D)`、CLS 特征 `(B,D)` 和分类 logits `(B,C)`；
7. 将模型前向扩展到 Dataset、DataLoader、loss、反向传播、优化器和 checkpoint 的完整训练系统；
8. 通过形状测试和单步更新测试逐层验证，而不是等完整训练失败后再排查。

更详细的 Python、PyTorch、ViT 形状推导和训练知识见 [PYTORCH_VIT_HANDBOOK.md](PYTORCH_VIT_HANDBOOK.md)。

## 已知限制与后续计划

当前版本以理解和正确性为优先，暂未实现：

- 预训练权重导入；
- 不同输入分辨率的位置编码插值；
- Linear Warmup；
- Mixup、CutMix、RandAugment；
- Stochastic Depth；
- 自动混合精度；
- TensorBoard/W&B 日志；
- 自动恢复中断训练的命令行入口；
- 多 GPU / DistributedDataParallel；
- 系统超参数搜索。

下一阶段计划：

1. 先用 16～32 张固定图片完成小数据过拟合检查；
2. 运行默认配置的完整训练并记录曲线和最佳结果；
3. 依次研究 warmup、label smoothing 和参数分组 weight decay；
4. 在保持可复现的前提下做单变量消融实验；
5. 对照官方实现继续检查初始化和训练细节。

## 参考资料

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- Google Research, [vision_transformer](https://github.com/google-research/vision_transformer)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)
- [Torchvision CIFAR-10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
