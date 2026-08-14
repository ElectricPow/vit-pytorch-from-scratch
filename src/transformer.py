import torch
from torch import nn

from src.patch_embedding import PatchEmbedding
from src.encoder import Encoder

class VisionTransformer(nn.Module):
    def __init__(self, 
                 image_size: int, patch_size: int, in_channels: int,
                 num_layers: int, embed_dim: int, 
                 num_heads: int, mlp_dim: int,
                 embed_dropout: float = 0.0, attention_dropout: float = 0.0, projection_dropout: float = 0.0,
                 mlp_dropout: float = 0.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.encoder = Encoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            mlp_dropout=mlp_dropout
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #(1,1,D)

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches + 1, embed_dim)) #(1,N+1,D)
        self.embed_dropout = nn.Dropout(embed_dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # image: (B,C,H,W)
        x = self.patch_embedding(x) #(B,N,D)

        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1) #(B,1,D)
        x = torch.cat([cls_tokens, x], dim=1) #(B,N+1,D)

        #Broadcast the position embedding to match the batch size
        x = x + self.pos_embedding #(B,N+1,D)
        x = self.embed_dropout(x) #(B,N+1,D)

        x = self.encoder(x) #(B,N+1,D)

        return x