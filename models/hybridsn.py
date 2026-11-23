import torch
import torch.nn as nn


class HybridSN(nn.Module):
    """
    HybridSN: 3D-CNN then 2D-CNN for hyperspectral classification.
    Roy et al., IEEE GRSL 2020.

    Constraint responses:
    - Spectral redundancy: 3D convolutions compress spectral-spatial jointly
    - Mixed pixels: local receptive field with spatial abstraction via 2D conv
    - Scalability: moderate parameter count, fast inference
    """

    def __init__(self, n_bands, n_classes, patch_size=13):
        super().__init__()
        self.n_bands    = n_bands
        self.n_classes  = n_classes
        self.patch_size = patch_size

        # ── 3D convolutional blocks ──────────────────────────────────────────
        # Input: (B, 1, C, P, P)
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Compute the spatial and spectral dimensions after 3D convs
        # to set up the 2D conv input correctly
        self._build_2d_block(n_bands, patch_size)

        # ── Classifier ──────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def _build_2d_block(self, n_bands, patch_size):
        """Compute 2D conv input shape via a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bands, patch_size, patch_size)
            x = self.conv3d_1(dummy)
            x = self.conv3d_2(x)
            x = self.conv3d_3(x)
            # Reshape: merge spectral and channel dims → (B, C*spec, H, W)
            b, c, s, h, w = x.shape
            self.reshape_channels = c * s
            x = x.reshape(b, self.reshape_channels, h, w)

        # ── 2D convolutional block ───────────────────────────────────────────
        self.conv2d = nn.Sequential(
            nn.Conv2d(self.reshape_channels, 64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Compute flat dimension after 2D conv
        with torch.no_grad():
            x = self.conv2d(x)
            self.flat_dim = x.numel() // x.shape[0]

    def forward(self, x):
        # x: (B, C, P, P) — add channel dim for 3D conv
        x = x.unsqueeze(1)                          # (B, 1, C, P, P)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)

        # Merge spectral and feature dims for 2D conv
        b, c, s, h, w = x.shape
        x = x.reshape(b, c * s, h, w)               # (B, C*S, H, W)
        x = self.conv2d(x)

        x = x.reshape(b, -1)                        # flatten
        x = self.classifier(x)
        return x


# ─── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    configs = [
        ('indian_pines', 30,  16, 13),
        ('pavia_u',      30,   9, 13),
        ('salinas',      30,  16, 13),
    ]

    for name, n_bands, n_classes, patch_size in configs:
        model = HybridSN(n_bands=n_bands, n_classes=n_classes,
                         patch_size=patch_size).to(device)
        x = torch.randn(4, n_bands, patch_size, patch_size).to(device)
        out = model(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{name}: input={tuple(x.shape)} '
              f'output={tuple(out.shape)} '
              f'params={params/1e3:.1f}K')

    print('\nhybridsn.py OK')