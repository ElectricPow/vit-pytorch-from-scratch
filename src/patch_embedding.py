import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, 
                 in_channels: int, embed_dim: int):
            super().__init__()

            # Input image:(B,C,H,W)

            self.image_size = image_size
            self.patch_size = patch_size
            self.in_channels = in_channels
            self.embed_dim = embed_dim

            self.grid_size = image_size // patch_size
            assert image_size % patch_size == 0, "Image size must be divisible by patch size"

            self.num_patches = self.grid_size ** 2

            #(B,C,H,W) -> (B,D,H/P,W/P)
            self.Conv = nn.Conv2d(in_channels=self.in_channels,
                                  out_channels=self.embed_dim,
                                  kernel_size=self.patch_size,
                                  stride=self.patch_size,
                                  padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
             # x: (B,C,H,W)
            x = self.Conv(x) #(B,D,H/P,W/P)
            x = x.flatten(2) #(B,D,N)
            x = x.transpose(1,2) #(B,N,D)

            return x