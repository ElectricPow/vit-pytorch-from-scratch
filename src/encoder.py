import torch
from torch import nn

from src.attention import MultiHeadSelfAttention
from src.mlp import MLP

class Encoder1Block(nn.Module):
    def __init__(self, 
                 embed_dim: int, num_heads: int, mlp_dim: int,
                 attention_dropout: float = 0.0, projection_dropout: float = 0.0,
                 mlp_dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim) #(B,N,D) -> (B,N,D)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads,
                                                attention_dropout,
                                                projection_dropout) #(B,N,D) -> (B,N,D)
        self.norm2 = nn.LayerNorm(embed_dim) #(B,N,D) -> (B,N,D)
        self.mlp = MLP(embed_dim, mlp_dim, mlp_dropout) #(B,N,D) -> (B,N,D) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D)
        x = x + self.attention(self.norm1(x)) #(B,N,D)
        x = x + self.mlp(self.norm2(x)) #(B,N,D)
        return x


class Encoder(nn.Module):
    def __init__(self, 
                 num_layers: int, embed_dim: int, num_heads: int, mlp_dim: int,
                 attention_dropout: float = 0.0, projection_dropout: float = 0.0,
                 mlp_dropout: float = 0.0):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"Number of layers ({num_layers}) must be at least 1")
        
        blocks =[]
        for _ in range(num_layers):
            block = Encoder1Block(embed_dim, num_heads, mlp_dim,
                                attention_dropout, projection_dropout,
                                mlp_dropout)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        #Layernorm after the last block (B,N,D) -> (B,N,D)
        self.norm = nn.LayerNorm(embed_dim) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D)
        for block in self.blocks:
            x = block(x) #(B,N,D)
        x = self.norm(x) #(B,N,D)
        return x