"""
ARCHIVED — legacy DRL agents retired in the boundary-tracing paradigm shift.

The active agent set is:
    DQN          → DQNAgent          (boundary tracing, patch network)
    DuelingDDQN  → DuelingDQNAgent   (boundary tracing, patch dueling network)
    DDPG         → DDPGAgent         (continuous mask-morphology baseline)

The classes below are no longer referenced by any config, notebook, or the
AGENT_REGISTRY:

    DDQNAgent            — plain Double-DQN. Superseded: DuelingDQNAgent already
                           combines double + dueling, and no config selects a
                           non-dueling double-DQN. Kept here for reference.
    MSADuelingDQNAgent   — Dueling DDQN with a Multi-Head Self-Attention backbone
                           (depends on archive/msa.py). Was the planned
                           "run MSA on the best base later" variant.

To resurrect either: move the class back into iteris/agents.py, re-add the
import, and register it in drl_training.AGENT_REGISTRY (and, for MSA, move
archive/msa.py back to iteris/msa.py).

This module is NOT imported by the package; it exists only as a recoverable
historical reference and is kept import-valid for that purpose.
"""

from copy import deepcopy

import torch

from ..agents import DQNAgent
from .msa     import MSADuelingQNetwork


class DDQNAgent(DQNAgent):
    """Double DQN (DQN with double=True)."""
    def __init__(self, **kw):
        kw['double']  = True
        kw['dueling'] = False
        super().__init__(**kw)


class MSADuelingDQNAgent(DQNAgent):
    """
    Dueling Double DQN with a Multi-Head Self-Attention backbone.

    The CNN global-average-pool is replaced by spatial-token self-attention
    (4 heads, 64-dim keys per head) before the Dueling Q head.  All training
    mechanics — Double DQN target, soft update, epsilon-greedy, Huber loss —
    are inherited unchanged from DQNAgent.
    """

    def __init__(
        self,
        in_channels: int   = 4,
        num_actions: int   = 13,
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        tau:         float = 0.005,
        embed_dim:   int   = 256,
        num_heads:   int   = 4,
        key_dim:     int   = 64,
        device:      torch.device = None,
    ):
        super().__init__(
            in_channels=in_channels,
            num_actions=num_actions,
            lr=lr,
            gamma=gamma,
            tau=tau,
            double=True,
            dueling=True,
            embed_dim=embed_dim,
            device=device,
        )
        self.q = MSADuelingQNetwork(
            in_channels, num_actions, embed_dim, num_heads, key_dim
        ).to(self.device)
        self.q_target = deepcopy(self.q).eval()
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
