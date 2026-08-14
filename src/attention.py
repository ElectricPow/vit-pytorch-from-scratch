import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 attention_dropout: float = 0.0, 
                 projection_dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim #D
        self.num_heads = num_heads #H
        self.head_dim = embed_dim // num_heads #d = D/H

        self.scale = self.head_dim ** -0.5 #1/sqrt(d)

        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3) #Q,K,V (B,N,D) -> (B,N,3D)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection = nn.Linear(embed_dim, embed_dim) #(B,N,D) -> (B,N,D)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D)
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x) #(B,N,3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim) #(B,N,3,H,d)
        qkv = qkv.permute(2, 0, 3, 1, 4) #(3,B,H,N,d)
        q, k, v = qkv.unbind(0) #(B,H,N,d)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) #(B,H,N,N)
        attn_scores = attn_scores * self.scale #(B,H,N,N)

        attention = torch.softmax(attn_scores, dim=-1) #(B,H,N,N)
        attention = self.attention_dropout(attention) #(B,H,N,N)

        # Compute weighted sum of values
        x = torch.matmul(attention, v) #(B,H,N,d)
        x = x.transpose(1, 2).reshape(B, N, D) #(B,N,D)

        # Projection
        x = self.projection(x) #(B,N,D)
        x = self.projection_dropout(x) #(B,N,D)

        return x
