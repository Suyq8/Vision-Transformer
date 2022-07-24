from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvolutionStem(nn.Module):
    def __init__(self, img_size, patch_size, in_chan, embed_dim):
        super().__init__()
        assert embed_dim % 8 == 0
        self.conv1 = Conv(in_chan, embed_dim//8, 3, 2)
        self.conv2 = Conv(embed_dim//8, embed_dim//4, 3, 2)
        self.conv3 = Conv(embed_dim//4, embed_dim//2, 3, 2)
        self.conv4 = Conv(embed_dim//2, embed_dim, 3, 2)
        self.conv5 = nn.Conv2d(embed_dim, embed_dim, 1)
        self.patch_size = patch_size
        self.h, self.w = img_size//patch_size, img_size//patch_size
        self.num_patch = self.h*self.w

        self.stem = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

    def forward(self, x):
        x = self.stem(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class PatchEnbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chan, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.h, self.w = img_size//patch_size, img_size//patch_size
        self.num_patch = self.h*self.w
        self.flatten = nn.Conv2d(in_chan, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.flatten(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_head=8, dropout_rate=0., bias=True):
        super().__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        dim_head = embed_dim//num_head
        self.scaler = 1/np.sqrt(dim_head)

        self.to_qkv = nn.Linear(embed_dim, embed_dim*3, bias=bias)
        nn.init.xavier_uniform_(self.to_qkv.weight)
        self.dropout = nn.Dropout(dropout_rate)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        qkv = self.to_qkv(x)
        # b, h, n, d
        q, k, v = rearrange(qkv, 'b n (c h d) -> c b h n d',
                            h=self.num_head, c=3).unbind(0)

        attention = q@k.transpose(2, 3)*self.scaler
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)  # b, h, n, n

        out = attention@v  # b, h, n, d
        out = out.transpose(1, 2)
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_head, mlp_dim, bias=True, dropout_rate=0.):
        super().__init__()
        self.attention = MultiheadAttention(
            embed_dim, num_head, dropout_rate, bias)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        layer1 = nn.Linear(embed_dim, mlp_dim)
        nn.init.xavier_uniform_(layer1.weight)
        nn.init.normal_(layer1.bias, std=1e-6)

        layer2 = nn.Linear(mlp_dim, embed_dim)
        nn.init.xavier_uniform_(layer2.weight)
        nn.init.normal_(layer2.bias, std=1e-6)

        self.mlp = nn.Sequential(
            layer1,
            nn.GELU(),
            nn.Dropout(dropout_rate),
            layer2,
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x+self.dropout(self.attention(self.norm1(x)))
        x = x+self.dropout(self.mlp(self.norm2(x)))

        return x


class viT(nn.Module):
    def __init__(self, img_size, patch_size, num_class, embed_dim, mlp_dim, num_encoder=5, num_head=8, in_chan=3, dropout_rate=0., global_pool=False, use_conv=False):
        super().__init__()
        self.num_class = num_class
        self.global_pool = global_pool
        if use_conv:
            self.patch_embedding = ConvolutionStem(
                img_size, patch_size, in_chan, embed_dim)
        else:
            self.patch_embedding = PatchEnbedding(
                img_size, patch_size, in_chan, embed_dim)
        self.num_patch = self.patch_embedding.num_patch

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.global_pool:
            self.pos_embed = nn.Parameter(torch.normal(
                0, 0.02, size=(1, self.num_patch, embed_dim)))
        else:
            self.pos_embed = nn.Parameter(torch.normal(
                0, 0.02, size=(1, self.num_patch+1, embed_dim)))
        self.dropout = nn.Dropout(dropout_rate)

        self.encoder = nn.Sequential(*[
            EncoderBlock(embed_dim, num_head, mlp_dim, dropout_rate=dropout_rate) for _ in range(num_encoder)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Linear(embed_dim, num_class)
        nn.init.zeros_(self.mlp_head.weight)
        nn.init.zeros_(self.mlp_head.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        if not self.global_pool:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)

        x = self.encoder(x)
        x = self.norm(x)

        x = x.mean(dim=1) if self.global_pool else x[:, 0]
        x = self.mlp_head(x)

        return x
