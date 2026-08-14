import torch
from src.attention import MultiHeadSelfAttention  

def test_MultiHeadSelfAttention():
    attention = MultiHeadSelfAttention(embed_dim=64, num_heads=8)

    tokens = torch.randn(2, 64, 64) #(B,N,D)
    output = attention(tokens) #(B,N,D)

    assert output.shape == (2, 64, 64) #(B,N,D)