"""
Network architectures for DRL agents.

Shared CNNBackbone consumes the (4, H, W) state and produces an embedding.
Five heads sit on top:
    QNetwork        — vanilla DQN
    DuelingQNetwork — V(s) + A(s,a) split with mean-centred advantage
    Actor           — DDPG actor with tanh-bounded continuous action
    Critic          — DDPG critic with mid-layer action injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_conv_stack(in_channels: int):
    """4-block stride-2 conv stack shared by CNNBackbone and the spatial heads.

    (B, in_channels, 256, 256) -> (B, 128, 16, 16) for the project's 256x256
    inputs (256->128->64->32->16 over the 4 stride-2 blocks). Factored out of
    CNNBackbone.__init__ verbatim (same ops, same order) so spatial modules can
    reuse it without duplicating code or touching CNNBackbone's behaviour.
    """
    ch = (32, 64, 128, 128)
    def block(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, oc),
            nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        block(in_channels, ch[0]),
        block(ch[0],       ch[1]),
        block(ch[1],       ch[2]),
        block(ch[2],       ch[3]),
    )


class CNNBackbone(nn.Module):
    """
    Stride-2 conv encoder: (B, in_C, H, W) → (B, embed_dim).

    Uses GroupNorm (not BatchNorm) — DRL trains on rollout batches that can have
    only a few unique sample indices, which makes BN statistics unstable.
    """

    def __init__(self, in_channels: int = 5, embed_dim: int = 256):
        super().__init__()
        self.conv = _make_conv_stack(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc   = nn.Linear(128 * 8 * 8, embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return F.relu(self.fc(x))



# PatchCNNBackbone, PatchQNetwork, PatchDuelingQNetwork were archived with
# Paradigm 1 (boundary tracing). See iteris/archive/paradigm1_boundary_tracing/.

class QNetwork(nn.Module):
    """Vanilla DQN Q-network."""
    def __init__(self, in_channels=5, num_actions=13, embed_dim=256):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN:  Q(s, a) = V(s) + A(s, a) − mean_a[A(s, a)]

    Mean-centred (NOT max-centred) — this is the form that keeps V(s) and
    A(s, a) identifiable. Using max produces V/A drift.
    """
    def __init__(self, in_channels=5, num_actions=13, embed_dim=256):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, embed_dim)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        emb = self.backbone(x)
        v   = self.value_head(emb)
        a   = self.adv_head(emb)
        return v + (a - a.mean(dim=1, keepdim=True))


class Actor(nn.Module):
    """
    DDPG actor — tanh output with per-component scaling.

    Outputs a 3-component action (mask-space DDPG):
        out[0] : morph   — SDT threshold shift (dilate / erode)
        out[1] : dy_norm — fractional y-translation
        out[2] : dx_norm — fractional x-translation

    ``action_scale`` is a list ``[morph_scale, trans_scale, trans_scale]``
    stored as a registered buffer so it moves with the model across devices.
    The tanh output (in [-1, +1]) is multiplied element-wise by action_scale,
    so each component lives in its own bounded range — morph and translation
    need different magnitudes (morph ≈ 0.25, trans ≈ 0.02 for 256-px images).

    Final-layer init is small-uniform [-3e-3, +3e-3] per Lillicrap et al. 2015
    to prevent tanh saturation during the actor-freeze warmup.
    """
    def __init__(self, in_channels=5, action_dim=3, action_scale=None, embed_dim=256):
        super().__init__()
        if action_scale is None:
            action_scale = [0.25, 0.02, 0.02]   # [morph, dy, dx] — tunable via YAML
        self.backbone = CNNBackbone(in_channels, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        # Per-component scale as a buffer: auto-moves with model.to(device)
        self.register_buffer(
            'action_scale',
            torch.tensor(action_scale, dtype=torch.float32),
        )
        # Small-uniform init on the FINAL layer only (standard DDPG practice)
        nn.init.uniform_(self.fc2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc2.bias,   -3e-3, 3e-3)

    def forward(self, x):
        h = F.relu(self.fc1(self.backbone(x)))
        return torch.tanh(self.fc2(h)) * self.action_scale   # element-wise scale


class Critic(nn.Module):
    """
    DDPG critic with mid-layer action injection.

    Accepts a 3-component action (morph, dy_norm, dx_norm).  The action is
    projected to 128-d and fused with the state embedding before the output
    head — this prevents the low-magnitude action signal from being drowned
    out by the high-dimensional state embedding at the final layer.

    Final-layer small-uniform init [-3e-3, +3e-3] keeps initial Q values near
    zero, preventing early actor exploitation of random Q spikes.
    """
    def __init__(self, in_channels=5, action_dim=3, embed_dim=256):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, embed_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, 128), nn.ReLU(inplace=True),
        )
        self.merge_fc1 = nn.Linear(embed_dim + 128, 128)
        self.merge_fc2 = nn.Linear(128, 1)
        nn.init.uniform_(self.merge_fc2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.merge_fc2.bias,   -3e-3, 3e-3)

    def forward(self, state, action):
        s = self.backbone(state)
        a = self.action_proj(action)
        h = F.relu(self.merge_fc1(torch.cat([s, a], dim=1)))
        return self.merge_fc2(h)


# ─── Spatial (per-sector local-feature) heads — additive, opt-in ──────────────
#
# Motivation: the Actor / DuelingQNetwork advantage head above predict ALL
# per-sector outputs from ONE globally-pooled embedding — the network has no
# way to localise "which output controls which angular wedge of the contour
# boundary". SectorPool gives each sector its own local receptive field
# (mean-pooled over the feature-map cells that fall in its angular wedge),
# while a parallel global-context branch keeps whole-image information
# available too ("where" + "what"). See module docstring for the angle
# convention this mirrors from env_contour_refine.py.

class SectorPool(nn.Module):
    """Angular-wedge mean-pooling over a conv feature map.

    Precomputes (once, as non-trainable buffers) a boolean-as-float assignment
    `sector_mask` of shape (n_sectors, feat_h*feat_w): cell (r, c) (flattened
    index r*feat_w + c) belongs to sector
        g = floor( mod(atan2(r - feat_h/2, c - feat_w/2), 2*pi) / (2*pi/n_sectors) )
    clipped to [0, n_sectors-1]. This is the EXACT angle convention
    ContourRefineEnv._sector_indices / _apply_continuous use (atan2(dy, dx),
    mod 2*pi, floor-binned) but centred at the feature map's own spatial centre
    instead of the live contour centroid — a static, zero-extra-data
    approximation valid because CAMUS/BRISC masks sit near the image centre
    after preprocessing.
    """

    def __init__(self, feat_h: int, feat_w: int, n_sectors: int):
        super().__init__()
        self.feat_h = int(feat_h)
        self.feat_w = int(feat_w)
        self.n_sectors = int(n_sectors)

        rr, cc = torch.meshgrid(
            torch.arange(self.feat_h, dtype=torch.float32),
            torch.arange(self.feat_w, dtype=torch.float32),
            indexing='ij',
        )
        dy = rr - self.feat_h / 2.0
        dx = cc - self.feat_w / 2.0
        ang = torch.atan2(dy, dx)                                   # (feat_h, feat_w)
        ang = torch.remainder(ang, 2.0 * torch.pi)
        wedge = (2.0 * torch.pi) / self.n_sectors
        bins = torch.floor(ang / wedge).long()
        bins = torch.clamp(bins, 0, self.n_sectors - 1)
        bins_flat = bins.reshape(-1)                                 # (feat_h*feat_w,)

        sector_mask = torch.zeros(self.n_sectors, self.feat_h * self.feat_w,
                                   dtype=torch.float32)
        sector_mask.scatter_(0, bins_flat.unsqueeze(0), 1.0)
        # ^ scatter_ along dim 0 writes a 1 at (bins_flat[i], i) for every i —
        #   equivalent to, for each sector g, sector_mask[g, i] = (bins_flat[i]==g).
        self.register_buffer('sector_mask', sector_mask)              # (n_sectors, H*W)
        self.register_buffer('sector_counts',
                              sector_mask.sum(dim=1).clamp(min=1.0))   # (n_sectors,)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """(B, C, feat_h, feat_w) -> (B, n_sectors, C) per-sector mean features."""
        b, c, h, w = feat_map.shape
        flat = feat_map.reshape(b, c, h * w)                         # (B, C, H*W)
        # sum over cells assigned to each sector: (B,C,H*W) x (n_sectors,H*W)^T
        summed = torch.einsum('bcn,gn->bgc', flat, self.sector_mask)  # (B, n_sectors, C)
        return summed / self.sector_counts.view(1, -1, 1)


def _global_context_branch(embed_dim_out: int = 64):
    """Small parallel global-context branch shared by the spatial heads:
    AdaptiveAvgPool2d(4) -> flatten -> Linear(128*4*4, embed_dim_out) -> ReLU.
    Operates on the same (B, 128, 16, 16) feature map the SectorPool consumes.
    """
    return nn.ModuleDict({
        'pool': nn.AdaptiveAvgPool2d(4),
        'fc':   nn.Linear(128 * 4 * 4, embed_dim_out),
    })


def _run_global_context(branch: nn.ModuleDict, feat_map: torch.Tensor) -> torch.Tensor:
    h = branch['pool'](feat_map).flatten(1)
    return F.relu(branch['fc'](h))


class SpatialActor(nn.Module):
    """
    TD3 actor with per-sector LOCAL features + a global context vector —
    mirrors Actor's public interface but gives each sector output its own
    receptive field instead of predicting all sectors from one pooled vector.

    n_sectors = action_dim (TD3's action_dim for the contour env IS the
    sector count — see ContourRefineEnv.CONTINUOUS_ACTION_DIM = cont_sectors).
    """

    def __init__(self, in_channels=5, action_dim=16, action_scale=None, embed_dim=256):
        super().__init__()
        self.n_sectors = int(action_dim)
        if action_scale is None:
            action_scale = [1.0] * self.n_sectors
        self.conv = _make_conv_stack(in_channels)            # -> (B,128,16,16)
        self.sector_pool = SectorPool(16, 16, self.n_sectors)
        self.global_branch = _global_context_branch(64)
        self.mlp = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.register_buffer(
            'action_scale',
            torch.tensor(action_scale, dtype=torch.float32),
        )
        # Small-uniform init on the FINAL linear layer only (same convention as Actor)
        final_linear = self.mlp[-1]
        nn.init.uniform_(final_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(final_linear.bias,   -3e-3, 3e-3)

    def forward(self, x):
        feat = self.conv(x)                                   # (B,128,16,16)
        local = self.sector_pool(feat)                         # (B,n_sectors,128)
        glob = _run_global_context(self.global_branch, feat)   # (B,64)
        glob_exp = glob.unsqueeze(1).expand(-1, self.n_sectors, -1)   # (B,n_sectors,64)
        h = torch.cat([local, glob_exp], dim=-1)                # (B,n_sectors,192)
        out = self.mlp(h).squeeze(-1)                           # (B,n_sectors)
        return torch.tanh(out) * self.action_scale


class SpatialCritic(nn.Module):
    """
    TD3 critic — generalized additive model over sectors: each sector's local
    feature + its own action scalar contributes independently to Q, plus a
    global term. Mirrors Critic's public interface.
    """

    def __init__(self, in_channels=5, action_dim=16, embed_dim=256):
        super().__init__()
        self.n_sectors = int(action_dim)
        self.conv = _make_conv_stack(in_channels)             # -> (B,128,16,16)
        self.sector_pool = SectorPool(16, 16, self.n_sectors)
        self.global_branch = _global_context_branch(64)
        self.sector_mlp = nn.Sequential(
            nn.Linear(128 + 1, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.global_value = nn.Linear(64, 1)
        for lin in (self.sector_mlp[-1], self.global_value):
            nn.init.uniform_(lin.weight, -3e-3, 3e-3)
            nn.init.uniform_(lin.bias,   -3e-3, 3e-3)

    def forward(self, state, action):
        feat = self.conv(state)                                # (B,128,16,16)
        local = self.sector_pool(feat)                          # (B,n_sectors,128)
        act = action.reshape(action.shape[0], self.n_sectors, 1)
        h = torch.cat([local, act], dim=-1)                     # (B,n_sectors,129)
        per_sector = self.sector_mlp(h).squeeze(-1)              # (B,n_sectors)
        glob = _run_global_context(self.global_branch, feat)    # (B,64)
        global_term = self.global_value(glob)                   # (B,1)
        return per_sector.sum(dim=1, keepdim=True) + global_term


class SpatialDuelingQNetwork(nn.Module):
    """
    Dueling DQN head specialised for the 18-action contour layout:
    8 push-OUT sectors + 8 push-IN sectors + smooth + stop
    (see ContourRefineEnv.SECTORS / NUM_DISCRETE_ACTIONS / DISCRETE_NAMES).

    Per-sector local features feed two shared-MLP heads (push-out, push-in);
    smooth/stop are global-only concepts and come from the global context
    vector alone. Combined exactly like the existing DuelingQNetwork:
    Q = V + (A - mean(A)).
    """

    N_SECTORS = 8   # hardcoded — this variant is built for the 18-action contour layout

    def __init__(self, in_channels=5, num_actions=18, embed_dim=256):
        super().__init__()
        if num_actions != 2 * self.N_SECTORS + 2:
            raise ValueError(
                f'SpatialDuelingQNetwork is specialised for the contour env\'s '
                f'18-action layout (2*{self.N_SECTORS}+2); got num_actions={num_actions}. '
                f'Use the plain DuelingQNetwork for other action counts.'
            )
        self.conv = _make_conv_stack(in_channels)              # -> (B,128,16,16)
        self.sector_pool = SectorPool(16, 16, self.N_SECTORS)
        self.global_branch = _global_context_branch(64)

        self.out_head = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.in_head = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        # smooth + stop: global-only concepts, not sector-local
        self.global_adv_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        feat = self.conv(x)                                     # (B,128,16,16)
        local = self.sector_pool(feat)                           # (B,8,128)
        glob = _run_global_context(self.global_branch, feat)    # (B,64)
        glob_exp = glob.unsqueeze(1).expand(-1, self.N_SECTORS, -1)   # (B,8,64)
        h = torch.cat([local, glob_exp], dim=-1)                 # (B,8,192)

        a_out = self.out_head(h).squeeze(-1)                      # (B,8)  actions 0..7
        a_in  = self.in_head(h).squeeze(-1)                       # (B,8)  actions 8..15
        a_global = self.global_adv_head(glob)                      # (B,2)  smooth=16, stop=17

        a = torch.cat([a_out, a_in, a_global], dim=1)              # (B,18) — matches
        # ContourRefineEnv.DISCRETE_NAMES ordering: out0..7, in0..7, smooth, stop
        v = self.value_head(glob)                                  # (B,1)
        return v + (a - a.mean(dim=1, keepdim=True))
