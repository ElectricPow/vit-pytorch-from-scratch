import torch
from src.encoder import Encoder1Block

def test_Encoder1Block():
    encoder_block = Encoder1Block(embed_dim=64, num_heads=8, mlp_dim=128)

    tokens = torch.randn(2, 64, 64) #(B,N,D)
    output = encoder_block(tokens) #(B,N,D)

    assert output.shape == (2, 64, 64) #(B,N,D)