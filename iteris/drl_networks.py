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


class CNNBackbone(nn.Module):
    """
    Stride-2 conv encoder: (B, in_C, H, W) → (B, embed_dim).

    Uses GroupNorm (not BatchNorm) — DRL trains on rollout batches that can have
    only a few unique sample indices, which makes BN statistics unstable.
    """

    def __init__(self, in_channels: int = 4, embed_dim: int = 256):
        super().__init__()
        ch = (32, 64, 128, 128)
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, oc),
                nn.ReLU(inplace=True),
            )
        self.conv = nn.Sequential(
            block(in_channels, ch[0]),
            block(ch[0],       ch[1]),
            block(ch[1],       ch[2]),
            block(ch[2],       ch[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc   = nn.Linear(ch[3] * 8 * 8, embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return F.relu(self.fc(x))


class QNetwork(nn.Module):
    """Vanilla DQN Q-network."""
    def __init__(self, in_channels=4, num_actions=7, embed_dim=256):
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
    def __init__(self, in_channels=4, num_actions=7, embed_dim=256):
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
    DDPG actor — tanh-bounded output scaled to ±action_scale.

    Final-layer init is small-uniform [-3e-3, +3e-3] per Lillicrap et al. 2015
    (the original DDPG paper). This is critical to prevent tanh saturation
    during the actor-freeze warmup — without it, default Kaiming init can let
    weights drift to where tanh outputs are pinned at ±1, causing mode collapse
    (the actor outputs the same maximal-magnitude action for every state).
    """
    def __init__(self, in_channels=4, action_dim=2, action_scale=0.04, embed_dim=256):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.action_scale = action_scale
        # Small-uniform init on the FINAL layer only (standard DDPG practice)
        nn.init.uniform_(self.fc2.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc2.bias,   -3e-3, 3e-3)

    def forward(self, x):
        h = F.relu(self.fc1(self.backbone(x)))
        return torch.tanh(self.fc2(h)) * self.action_scale


class Critic(nn.Module):
    """
    DDPG critic with mid-layer action injection.

    The action is projected to 128-d and concatenated with the state embedding
    BEFORE the final layers. Concatenating a 2-d action to a final 256-d state
    embedding (as the project spec originally implied) means the action gets
    drowned out and the critic learns a near-state-only Q.

    Final-layer small-uniform init [-3e-3, +3e-3] keeps initial Q estimates
    near zero — prevents the actor from exploiting random Q spikes during
    early training.
    """
    def __init__(self, in_channels=4, action_dim=2, embed_dim=256):
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
