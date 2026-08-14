import torch
from src.mlp import MLP

def test_MLP():
    mlp = MLP(embed_dim=64, mlp_dim=128)

    tokens = torch.randn(2, 64, 64) #(B,N,D)
    output = mlp(tokens) #(B,N,D)

    assert output.shape == (2, 64, 64) #(B,N,D)