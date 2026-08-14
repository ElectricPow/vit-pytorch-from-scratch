import torch
from src.patch_embedding import PatchEmbedding

def test_PatchEmbedding():

    patch_embedding = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=64)

    images = torch.randn(2,3,32,32) #(B,C,H,W)
    tokens = patch_embedding(images) #(B,N,D)

    assert tokens.shape == (2, 64, 64) #(B,N,D)

