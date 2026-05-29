"""
DRL agents.

DQNAgent : vanilla / Double / Dueling DQN — flags toggle behaviour
DDPGAgent : continuous-action actor-critic with mid-layer action injection,
            OU exploration noise, and a 2k-step actor freeze window.

All agents accept a `state_builder(idx, mask) → torch tensor (4, H, W)` callable
so the buffer can store compact transitions and reconstruct full states at
update time.
"""

from copy import deepcopy
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .drl_networks import QNetwork, DuelingQNetwork, Actor, Critic
from .msa         import MSADuelingQNetwork


def _soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    for p, p_t in zip(source_net.parameters(), target_net.parameters()):
        p_t.data.mul_(1.0 - tau).add_(tau * p.data)


# ─── DQN family ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN (vanilla / Double / Dueling) for discrete-action boundary refinement.

    The discrete action space has 13 actions in env v3 (4 directional dilate +
    4 directional erode + 4 whole-mask shifts + no-op).  ``num_actions`` is
    set by the training loop from ``SegmentationEnv.NUM_DISCRETE_ACTIONS``.

    Toggle behaviour via:
        double=True   → Double DQN target computation
        dueling=True  → use DuelingQNetwork instead of QNetwork
    DDQNAgent and DuelingDQNAgent are thin subclasses that set these.
    """
    action_type = 'discrete'

    def __init__(
        self,
        in_channels: int = 4,
        num_actions: int = 13,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        double: bool = False,
        dueling: bool = False,
        embed_dim: int = 256,
        device: torch.device = None,
    ):
        self.device      = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_actions = num_actions
        self.gamma       = gamma
        self.tau         = tau
        self.double      = double
        self.dueling     = dueling

        NetCls = DuelingQNetwork if dueling else QNetwork
        self.q        = NetCls(in_channels, num_actions, embed_dim).to(self.device)
        self.q_target = deepcopy(self.q).eval()
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(self.num_actions))
        s = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        return int(self.q(s).argmax(dim=1).item())

    def update(self, batch: dict, state_builder: Callable) -> dict:
        n = len(batch['sample_idx'])
        cs = batch.get('current_sdt')           # may be None if buffer.cache_sdt=False
        ns = batch.get('next_sdt')
        states = torch.stack([
            state_builder(batch['sample_idx'][i], batch['current_mask'][i],
                          cs[i] if cs is not None else None)
            for i in range(n)
        ]).to(self.device)
        next_states = torch.stack([
            state_builder(batch['sample_idx'][i], batch['next_mask'][i],
                          ns[i] if ns is not None else None)
            for i in range(n)
        ]).to(self.device)
        actions = torch.from_numpy(batch['action']).long().to(self.device)
        rewards = torch.from_numpy(batch['reward']).to(self.device)
        dones   = torch.from_numpy(batch['done']).float().to(self.device)

        # Predict Q(s, a)
        q_pred = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target — Double DQN uses online net for action selection
        with torch.no_grad():
            if self.double:
                next_a = self.q(next_states).argmax(dim=1, keepdim=True)
                q_next = self.q_target(next_states).gather(1, next_a).squeeze(1)
            else:
                q_next = self.q_target(next_states).max(dim=1).values
            y = rewards + self.gamma * q_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_pred, y)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        _soft_update(self.q_target, self.q, self.tau)
        return {'loss': float(loss.item())}

    def state_dict(self):
        return {'q': self.q.state_dict(), 'q_target': self.q_target.state_dict()}

    def load_state_dict(self, state):
        self.q.load_state_dict(state['q'])
        self.q_target.load_state_dict(state['q_target'])


class DDQNAgent(DQNAgent):
    """Double DQN (DQN with double=True)."""
    def __init__(self, **kw):
        kw['double']  = True
        kw['dueling'] = False
        super().__init__(**kw)


class DuelingDQNAgent(DQNAgent):
    """Dueling Double DQN (best-practice combo: dueling head + double-DQN target)."""
    def __init__(self, **kw):
        kw['double']  = True
        kw['dueling'] = True
        super().__init__(**kw)


# ─── DDPG ──────────────────────────────────────────────────────────────────────

class OUNoise:
    """
    Discrete-time Ornstein–Uhlenbeck exploration noise with per-component sigma.

    ``sigma`` can be a scalar (uniform across all components) or a list/array
    with one value per action dimension.  Per-component sigma is essential when
    action components have different scales — e.g., morph (±0.25) vs.
    translation (±0.02) require different noise magnitudes for proportional
    exploration coverage.
    """
    def __init__(self, action_dim: int, theta: float = 0.15, sigma = 0.01):
        self.action_dim = action_dim
        self.theta      = theta
        # Broadcast scalar or list → (action_dim,) float32 array
        sig = np.atleast_1d(np.asarray(sigma, dtype=np.float32))
        self.sigma = np.broadcast_to(sig, (action_dim,)).copy()
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def sample(self) -> np.ndarray:
        self.state += (
            -self.theta * self.state
            + self.sigma * np.random.randn(self.action_dim).astype(np.float32)
        )
        return self.state.copy()


class DDPGAgent:
    """
    DDPG with mid-layer action injection, actor-freeze warmup, OU noise.

    The continuous action is 3-component: (morph, dy_norm, dx_norm).
    ``action_scale`` is a list ``[morph_scale, trans_scale, trans_scale]``
    that sets the per-component output range of the actor (tanh × scale).
    ``ou_sigma`` can be a scalar (same noise for all components) or a list
    with per-component values — use a list to keep noise proportional to
    each component's range (e.g., [0.025, 0.002, 0.002]).
    """
    action_type = 'continuous'

    def __init__(
        self,
        in_channels:        int   = 4,
        action_dim:         int   = 3,
        action_scale              = None,   # list [morph_scale, trans_scale, trans_scale]
        actor_lr:           float = 1e-4,
        critic_lr:          float = 1e-3,
        gamma:              float = 0.99,
        tau:                float = 0.005,
        ou_theta:           float = 0.15,
        ou_sigma                  = None,   # scalar or per-component list
        actor_freeze_steps: int   = 2000,
        embed_dim:          int   = 256,
        device:             torch.device = None,
    ):
        self.device             = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_dim         = action_dim
        self.gamma              = gamma
        self.tau                = tau
        self.actor_freeze_steps = actor_freeze_steps

        if action_scale is None:
            action_scale = [0.25, 0.02, 0.02]   # [morph, dy, dx]
        self.action_scale_np = np.array(action_scale, dtype=np.float32)

        # Default ou_sigma: 10 % of each component's range for proportional exploration
        if ou_sigma is None:
            ou_sigma = (self.action_scale_np * 0.1).tolist()

        self.actor         = Actor(in_channels, action_dim, action_scale, embed_dim).to(self.device)
        self.critic        = Critic(in_channels, action_dim, embed_dim).to(self.device)
        self.actor_target  = deepcopy(self.actor).eval()
        self.critic_target = deepcopy(self.critic).eval()
        for p in self.actor_target.parameters():  p.requires_grad_(False)
        for p in self.critic_target.parameters(): p.requires_grad_(False)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.noise      = OUNoise(action_dim, theta=ou_theta, sigma=ou_sigma)
        self.step_count = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        s = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        a = self.actor(s).cpu().numpy().squeeze(0)   # already scaled by action_scale buffer
        if explore:
            a = a + self.noise.sample()
        # Clip per-component to ±action_scale
        return np.clip(a, -self.action_scale_np, self.action_scale_np).astype(np.float32)

    def update(self, batch: dict, state_builder: Callable) -> dict:
        n = len(batch['sample_idx'])
        cs = batch.get('current_sdt')
        ns = batch.get('next_sdt')
        states = torch.stack([
            state_builder(batch['sample_idx'][i], batch['current_mask'][i],
                          cs[i] if cs is not None else None)
            for i in range(n)
        ]).to(self.device)
        next_states = torch.stack([
            state_builder(batch['sample_idx'][i], batch['next_mask'][i],
                          ns[i] if ns is not None else None)
            for i in range(n)
        ]).to(self.device)
        actions = torch.from_numpy(batch['action']).float().to(self.device)
        rewards = torch.from_numpy(batch['reward']).to(self.device)
        dones   = torch.from_numpy(batch['done']).float().to(self.device)

        # Critic update
        with torch.no_grad():
            next_a = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_a).squeeze(1)
            y      = rewards + self.gamma * q_next * (1.0 - dones)

        q_pred = self.critic(states, actions).squeeze(1)
        critic_loss = F.smooth_l1_loss(q_pred, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss_val = 0.0
        if self.step_count >= self.actor_freeze_steps:
            a_pred = self.actor(states)
            actor_loss = -self.critic(states, a_pred).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            actor_loss_val = float(actor_loss.item())

            _soft_update(self.actor_target,  self.actor,  self.tau)
            _soft_update(self.critic_target, self.critic, self.tau)

        self.step_count += 1
        return {'critic_loss': float(critic_loss.item()), 'actor_loss': actor_loss_val}

    def state_dict(self):
        return {
            'actor':         self.actor.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic':        self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])


# ─── MSA variants ──────────────────────────────────────────────────────────────



class MSADuelingDQNAgent(DQNAgent):
    """
    Dueling Double DQN with a Multi-Head Self-Attention backbone.

    The CNN global-average-pool is replaced by spatial-token self-attention
    (4 heads, 64-dim keys per head) before the Dueling Q head.  All training
    mechanics — Double DQN target, soft update, epsilon-greedy, Huber loss —
    are inherited unchanged from DQNAgent.

    Strategy: call DQNAgent.__init__ with dueling=True / double=True (which
    wires up all attributes), then swap self.q / self.q_target for MSA-backed
    networks and rebind the optimizer.  No training code is duplicated.

    Args
    ----
    num_heads : int   Number of attention heads (default 4, per CONTEXT.md).
    key_dim   : int   Per-head key/query/value dimension (default 64).
    All other args mirror DQNAgent.
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
        # Initialise base agent (creates DuelingQNetwork internally — we replace it next)
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
        # Swap in MSA-backed networks
        self.q = MSADuelingQNetwork(
            in_channels, num_actions, embed_dim, num_heads, key_dim
        ).to(self.device)
        self.q_target = deepcopy(self.q).eval()
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        # Rebind optimizer to the new network's parameters
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)


