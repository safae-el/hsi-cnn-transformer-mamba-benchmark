import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GroupWiseSpectralEmbedding(nn.Module):
    """
    GSE: learns local spectral relations from groups of contiguous bands.
    Hong et al., SpectralFormer, IEEE TGRS 2022.
    """
    def __init__(self, n_bands, d_model, band_patches=3):
        super().__init__()
        self.band_patches = band_patches
        # Each group of band_patches bands → one token of size d_model
        self.embedding = nn.Linear(band_patches, d_model)
        # How many tokens per pixel
        self.n_tokens = n_bands // band_patches

    def forward(self, x):
        # x: (B, C, P, P) — we work on the spectral dimension
        B, C, H, W = x.shape
        # Reshape spatial patches into pixel vectors
        x = x.permute(0, 2, 3, 1)           # (B, H, W, C)
        x = x.reshape(B * H * W, C)         # (N, C)

        # Group bands
        n_tokens = C // self.band_patches
        x = x[:, :n_tokens * self.band_patches]
        x = x.reshape(-1, n_tokens, self.band_patches)  # (N, T, bp)
        x = self.embedding(x)                            # (N, T, d_model)
        return x, B, H, W, n_tokens


class CrossLayerAdaptiveFusion(nn.Module):
    """
    CAF: residual fusion between transformer layers.
    Carries spectral memory across encoder blocks.
    """
    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, current, previous):
        return self.alpha * current + (1 - self.alpha) * previous


class SpectralTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
        self.caf = CrossLayerAdaptiveFusion(d_model)

    def forward(self, x, prev=None):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # CAF fusion with previous layer output
        if prev is not None:
            x = self.caf(x, prev)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralFormer(nn.Module):
    """
    SpectralFormer: group-wise spectral transformer for HSI classification.
    Hong et al., IEEE TGRS 2022.

    Constraint responses:
    - Spectral redundancy: grouped band attention learns inter-band relations
    - Mixed pixels: patch-wise mode captures local spatial context
    - Limited labels: attention-based pretraining potential
    - Scalability: lightweight compared to ViT — ~400K params
    """

    def __init__(self, n_bands, n_classes, patch_size=13,
                 d_model=64, n_heads=4, n_layers=5,
                 band_patches=3, dropout=0.1):
        super().__init__()
        self.patch_size  = patch_size
        self.n_classes   = n_classes

        # Spectral embedding
        self.gse = GroupWiseSpectralEmbedding(n_bands, d_model, band_patches)
        n_tokens = n_bands // band_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            SpectralTransformerBlock(d_model, n_heads,
                                     mlp_dim=d_model * 4,
                                     dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Spatial aggregation for patch-wise mode
        # After spectral transformer we have per-pixel tokens
        # We aggregate spatially using a small conv
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, P, P)
        B, C, H, W = x.shape

        # Spectral embedding → tokens per pixel
        tokens, B, H, W, n_tokens = self.gse(x)  # (B*H*W, T, d_model)
        N = B * H * W

        # Prepend CLS token
        cls = self.cls_token.expand(N, -1, -1)    # (N, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)   # (N, T+1, d_model)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        tokens = self.dropout(tokens)

        # Transformer blocks with CAF
        prev = None
        for block in self.blocks:
            tokens = block(tokens, prev)
            prev = tokens

        tokens = self.norm(tokens)

        # Use CLS token as pixel representation
        cls_out = tokens[:, 0]                     # (N, d_model)

        # Reshape back to spatial grid
        cls_out = cls_out.reshape(B, H, W, -1)     # (B, H, W, d_model)
        cls_out = cls_out.permute(0, 3, 1, 2)      # (B, d_model, H, W)

        # Spatial pooling to get single vector per sample
        cls_out = self.spatial_pool(cls_out)        # (B, d_model, 1, 1)
        cls_out = cls_out.reshape(B, -1)            # (B, d_model)

        return self.head(cls_out)                   # (B, n_classes)


# ─── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    configs = [
        ('indian_pines', 30, 16, 13),
        ('pavia_u',      30,  9, 13),
        ('salinas',      30, 16, 13),
    ]

    for name, n_bands, n_classes, patch_size in configs:
        model = SpectralFormer(
            n_bands=n_bands, n_classes=n_classes,
            patch_size=patch_size, d_model=64,
            n_heads=4, n_layers=5, band_patches=3
        ).to(device)

        x = torch.randn(4, n_bands, patch_size, patch_size).to(device)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()
                     if p.requires_grad)
        print(f'{name}: input={tuple(x.shape)} '
              f'output={tuple(out.shape)} '
              f'params={params/1e3:.1f}K')

    print('\nspectralformer.py OK')