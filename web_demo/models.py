"""
models.py — Backbone architectures cho training pipeline

Hai kiến trúc có sẵn, interface giống nhau:
  BackboneCNN      : Conv1D × 3 + AvgPool  (baseline)
  BackboneConvLSTM : Conv1D × 2 + LSTM     (cải tiến temporal)

Input:  (batch, 9 channels, 100 timesteps)
Output: (batch, 128) embedding  — giữ nguyên cho cả hai
"""
import torch
import torch.nn as nn


EMBED_DIM  = 128
N_CHANNELS = 9
WINDOW_LEN = 100


# ── Kiến trúc 1: CNN thuần (baseline) ──────────────────────────────────

class BackboneCNN(nn.Module):

    def __init__(self, n_users: int, n_channels: int = N_CHANNELS,
                 embed_dim: int = EMBED_DIM, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, n_users)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(self.dropout(emb))
        return logits, emb


# ── Kiến trúc 2: Conv + LSTM (DeepConvLSTM-style) ──────────────────────

class _ConvLSTMEncoder(nn.Module):
    """Conv × 2 giảm sequence 100→25, LSTM học temporal dependencies dài hạn.

    Flow: (B,9,100) → conv → (B,128,25) → transpose → (B,25,128)
          → LSTM → last hidden → (B,embed_dim)
    """

    def __init__(self, n_channels: int, embed_dim: int,
                 dropout: float, bidirectional: bool):
        super().__init__()
        # Với BiLSTM: mỗi chiều hidden_size = embed_dim//2 → concat = embed_dim
        lstm_hidden = embed_dim // 2 if bidirectional else embed_dim

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)            # (B, 128, 25)
        x = x.permute(0, 2, 1)     # (B, 25, 128)
        _, (h, _) = self.lstm(x)   # h: (D, B, hidden)  D=1 uni / D=2 bi
        if h.shape[0] > 1:         # bidirectional → cat forward + backward
            h = torch.cat([h[0], h[1]], dim=-1)
        else:
            h = h.squeeze(0)
        return self.drop(h)         # (B, embed_dim)


class BackboneConvLSTM(nn.Module):
    """Conv1D × 2 + LSTM — học tốt hơn pattern tuần tự dài hạn (gait cycle).

    Cùng interface với BackboneCNN: forward() trả (logits, embedding).
    """

    def __init__(self, n_users: int, n_channels: int = N_CHANNELS,
                 embed_dim: int = EMBED_DIM, dropout: float = 0.4,
                 bidirectional: bool = False):
        super().__init__()
        self.encoder = _ConvLSTMEncoder(n_channels, embed_dim, dropout, bidirectional)
        self.classifier = nn.Linear(embed_dim, n_users)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, emb


# ── Lookup helper ───────────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    'cnn':         BackboneCNN,
    'convlstm':    BackboneConvLSTM,
    'convlstm_bi': lambda **kw: BackboneConvLSTM(bidirectional=True, **kw),
}


def build_backbone(arch: str, n_users: int, **kwargs) -> nn.Module:
    """Khởi tạo backbone theo tên: 'cnn' | 'convlstm' | 'convlstm_bi'."""
    if arch not in BACKBONE_REGISTRY:
        raise ValueError(f"arch phải là {list(BACKBONE_REGISTRY)}, nhận: {arch!r}")
    return BACKBONE_REGISTRY[arch](n_users=n_users, **kwargs)


# ── Load checkpoint ─────────────────────────────────────────────────────

def load_encoder(checkpoint_path: str, n_users: int = 19,
                 arch: str = 'cnn') -> nn.Module:
    """Load checkpoint và trả về encoder-only (không có classifier head).

    arch : 'cnn' | 'convlstm' | 'convlstm_bi' — phải khớp với lúc train.
    """
    model = build_backbone(arch, n_users=n_users)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    ckpt_cls = state.get('classifier.weight')
    if ckpt_cls is not None and ckpt_cls.shape[0] != n_users:
        print(f"[load_encoder] classifier shape mismatch "
              f"(ckpt={ckpt_cls.shape[0]}, current={n_users}); "
              f"dropping classifier weights.")
        state = {k: v for k, v in state.items() if not k.startswith('classifier')}

    model.load_state_dict(state, strict=False)
    model.eval()

    class EncoderOnly(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.encoder = backbone.encoder
        def forward(self, x):
            return self.encoder(x)

    return EncoderOnly(model).eval()
