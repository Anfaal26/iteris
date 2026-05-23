"""
Multi-Head Self-Attention (MSA) modules for DRL agents.

MSABackbone replaces CNNBackbone's global-average-pool with spatial-token
self-attention, giving the agent explicit cross-position reasoning over
the 8×8 feature map produced by the strided conv stack.

Architecture:
    (B, 4, 256, 256)
    → 4× stride-2 conv blocks (GroupNorm + ReLU)   (B, 128, 16, 16)
    → AdaptiveAvgPool2d(8)                          (B, 128,  8,  8)
    → reshape to spatial tokens                     (B, 64, 128)
    → linear projection to attn_dim                 (B, 64, attn_dim)   [attn_dim = num_heads × key_dim]
    → + learnable positional encoding
    → Multi-Head Self-Attention                     (B, 64, attn_dim)
    → residual + LayerNorm
    → mean-pool over tokens                         (B, attn_dim)
    → linear → ReLU                                 (B, embed_dim)

Two MSA-specific network heads:
    MSADuelingQNetwork — Dueling DQN head on MSABackbone
    MSAActor           — DDPG actor head on MSABackbone

Per CONTEXT.md spec: "4-head MSA, 64-dim keys" (attn_dim = 4 × 64 = 256).
The standard CNN-backed Critic from drl_networks.py is reused unchanged
for MSA-DDPG — the spec says "MSA in actor" only, and the critic's
state–action interface is identical regardless of which backbone the
actor uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSABackbone(nn.Module):
    """
    CNN backbone with Multi-Head Self-Attention on spatial tokens.

    Replaces the flat global-average-pool of CNNBackbone with explicit
    self-attention over the 8×8 spatial grid (64 tokens), then mean-pools
    the attended tokens into a fixed-size embedding.

    Same conv stack as CNNBackbone (4× stride-2 blocks, GroupNorm) so the
    receptive field and parameter count are comparable — the only addition
    is the MSA layer and positional encoding.

    Args
    ----
    in_channels : int   State channels (4: image, mask, SDT, init_mask).
    embed_dim   : int   Final embedding dimension fed to the decision head.
    num_heads   : int   Number of parallel attention heads.
    key_dim     : int   Dimension of each head's key / query / value vectors.
                        attn_dim = num_heads × key_dim = 4 × 64 = 256 (default).
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim:   int = 256,
        num_heads:   int = 4,
        key_dim:     int = 64,
    ):
        super().__init__()
        ch        = (32, 64, 128, 128)
        token_dim = ch[-1]               # 128 — spatial token dim before projection
        attn_dim  = num_heads * key_dim  # 256 — attention working dimension

        def _block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, oc),
                nn.ReLU(inplace=True),
            )

        self.conv = nn.Sequential(
            _block(in_channels, ch[0]),   # (B, 32,  128, 128)
            _block(ch[0],       ch[1]),   # (B, 64,   64,  64)
            _block(ch[1],       ch[2]),   # (B, 128,  32,  32)
            _block(ch[2],       ch[3]),   # (B, 128,  16,  16)
        )
        self.pool = nn.AdaptiveAvgPool2d(8)   # → (B, 128, 8, 8) → 64 tokens

        # Project 128-d CNN tokens up to attention working dimension
        self.token_proj = nn.Linear(token_dim, attn_dim)

        # Learnable positional encoding for all 64 spatial positions (8×8 grid).
        # Small init — content should dominate position at training start.
        self.pos_enc = nn.Parameter(torch.randn(1, 64, attn_dim) * 0.02)

        # 4-head self-attention.  batch_first=True so shapes are (B, seq, dim).
        self.msa = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(attn_dim)

        # Final projection to embed_dim (same interface as CNNBackbone.fc)
        self.fc = nn.Linear(attn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.conv(x)                             # (B, 128, 16, 16)
        x = self.pool(x)                             # (B, 128,  8,  8)

        # Rearrange spatial feature map → sequence of tokens
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)        # (B, 64, 128)

        # Project to attention dimension and inject positional encoding
        tokens = self.token_proj(tokens)             # (B, 64, 256)
        tokens = tokens + self.pos_enc               # broadcast over batch dim

        # Multi-Head Self-Attention with pre-norm residual
        attended, _ = self.msa(tokens, tokens, tokens)   # (B, 64, 256)
        tokens = self.norm(tokens + attended)             # residual + LayerNorm

        # Aggregate over spatial tokens → single embedding vector
        pooled = tokens.mean(dim=1)                  # (B, 256)
        return F.relu(self.fc(pooled))               # (B, embed_dim)


class MSADuelingQNetwork(nn.Module):
    """
    Dueling DQN Q-network with MSA backbone.

    Q(s, a) = V(s) + A(s, a) − mean_a[A(s, a)]   (mean-centred aggregation,
    same formula as DuelingQNetwork — keeps V and A identifiable).

    Drop-in replacement for DuelingQNetwork — identical forward interface
    (state tensor in, per-action Q-values out).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_actions: int = 7,
        embed_dim:   int = 256,
        num_heads:   int = 4,
        key_dim:     int = 64,
    ):
        super().__init__()
        self.backbone = MSABackbone(in_channels, embed_dim, num_heads, key_dim)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        v   = self.value_head(emb)
        a   = self.adv_head(emb)
        return v + (a - a.mean(dim=1, keepdim=True))


class MSAActor(nn.Module):
    """
    DDPG actor with MSA backbone — drop-in replacement for Actor.

    Outputs a 3-component action (morph, dy_norm, dx_norm) with per-component
    scaling stored as a registered buffer (mirrors the CNN Actor interface).
    The MSA backbone gives the actor explicit cross-position attention over the
    8×8 spatial token grid before computing the action — better boundary
    awareness than the plain CNN Actor's global-average-pool.

    Retains the DDPG convention of small-uniform final-layer init [-3e-3, +3e-3]
    to prevent tanh saturation during the actor-freeze warmup period.
    """

    def __init__(
        self,
        in_channels:  int   = 4,
        action_dim:   int   = 3,
        action_scale        = None,   # list [morph_scale, trans_scale, trans_scale]
        embed_dim:    int   = 256,
        num_heads:    int   = 4,
        key_dim:      int   = 64,
    ):
        super().__init__()
        if action_scale is None:
            action_scale = [0.25, 0.02, 0.02]   # [morph, dy, dx]
        self.backbone = MSABackbone(in_channels, embed_dim, num_heads, key_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        # Per-component scale as a buffer — auto-moves with model.to(device)
        self.register_buffer(
            'action_scale',
            torch.tensor(action_scale, dtype=torch.float32),
        )
        # Small-uniform init on the output layer only (standard DDPG practice)
        nn.init.uniform_(self.fc2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc2.bias,   -3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self.backbone(x)))
        return torch.tanh(self.fc2(h)) * self.action_scale   # element-wise scale
