import torch
import torch.nn as nn
from einops import rearrange, repeat
import math


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        self.qkv_linear = nn.Linear(d_model, d_model * 3)
        nn.init.xavier_uniform_(self.qkv_linear.weight)

        self.fc = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, mask=None):
        qkv = self.qkv_linear(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h k) -> b h l k', h = self.n_heads), qkv)

        attn = torch.einsum('b h l k, b h t k -> b h l t', q, k) / q.shape[-1]**0.5

        if mask is not None:
            attn = attn.masked_fill(mask[None], -float('inf'))
        
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.einsum('b h l t, b h t v -> b h l v', [attn, v])
        output = rearrange(output, 'b h l v -> b l (h v)')
        output = self.dropout(self.fc(output))
        return output, attn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network

        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.feed_forword = FeedForward(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x)[0]
        x = x + self.feed_forword(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.patch_to_embedding = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        cls = x[:, 0]
        return self.mlp_head(cls)