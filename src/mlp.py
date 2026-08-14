import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, mlp_dim) #(B,N,D) -> (B,N,mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(mlp_dim, embed_dim) #(B,N,mlp_dim) -> (B,N,D)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,D)
        x = self.fc1(x) #(B,N,D) -> (B,N,mlp_dim)
        x = self.act(x) #(B,N,mlp_dim)
        x = self.dropout1(x) #(B,N,mlp_dim)

        x = self.fc2(x) #(B,N,mlp_dim) -> (B,N,D)
        x = self.dropout2(x) #(B,N,D)

        return x