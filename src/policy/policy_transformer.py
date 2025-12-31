import torch
import torch.nn as nn


class PolicyTransformer(nn.Module):
    def __init__(self, token_dim, n_layers=4, n_heads=8, dim_ff=512, dropout=0.1, out_dim=512):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(token_dim, out_dim)

    def forward(self, tokens, attention_mask=None):
        # tokens: (B, T, token_dim)
        x = self.transformer(tokens)  # (B, T, token_dim)
        # pool over tokens (mean)
        x = x.mean(dim=1)  # (B, token_dim)
        out = self.proj(x)  # (B, out_dim)
        return out

    def generate_action(self, x):
        x = self.forward(x)
        action = torch.argmax(x, dim=-1)
        return action