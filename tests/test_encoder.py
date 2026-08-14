import torch
from src.encoder import Encoder1Block
from src.encoder import Encoder

def test_Encoder1Block():
    encoder_block = Encoder1Block(embed_dim=64, num_heads=8, mlp_dim=128)

    tokens = torch.randn(2, 64, 64) #(B,N,D)
    output = encoder_block(tokens) #(B,N,D)

    assert output.shape == (2, 64, 64) #(B,N,D)


def test_Encoder():
    model = Encoder(
        num_layers=4,
        embed_dim=64,
        num_heads=8,
        mlp_dim=256,
    )

    x = torch.randn(2, 65, 64)
    output = model(x)

    assert output.shape == (2, 65, 64)
    assert len(model.blocks) == 4

    # 两个列表元素必须是不同的 Encoder1Block 对象
    assert model.blocks[0] is not model.blocks[1]

    weight0 = model.blocks[0].attention.qkv.weight
    weight1 = model.blocks[1].attention.qkv.weight

    # 两个 Parameter 对象不同，底层存储地址也不同
    assert weight0 is not weight1
    assert weight0.data_ptr() != weight1.data_ptr()