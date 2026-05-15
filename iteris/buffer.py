"""
Replay buffer for DRL agents.

Memory-optimised storage: instead of full (4, H, W) states, we store the
sample_idx + binary mask (uint8) per transition. The full state is reconstructed
at sample time by combining cached static channels (image, init_mask) with the
dynamic mask and a freshly-computed SDT.

For 10k transitions at 256x256 this reduces buffer memory from ~20 GB to ~1.3 GB.
"""

from typing import Dict
import numpy as np


class ReplayBuffer:
    """Uniform circular replay buffer."""

    def __init__(
        self,
        capacity: int,
        mask_shape: tuple,
        action_dim: int = None,
        discrete: bool = True,
    ):
        self.capacity   = capacity
        self.mask_shape = mask_shape
        self.discrete   = discrete

        # Static reference per transition
        self.sample_idx   = np.zeros(capacity, dtype=np.int32)
        # Dynamic state at t and t+1
        self.current_mask = np.zeros((capacity, *mask_shape), dtype=np.uint8)
        self.next_mask    = np.zeros((capacity, *mask_shape), dtype=np.uint8)
        self.reward       = np.zeros(capacity, dtype=np.float32)
        self.done         = np.zeros(capacity, dtype=np.uint8)

        if discrete:
            self.action = np.zeros(capacity, dtype=np.int64)
        else:
            if action_dim is None:
                raise ValueError('action_dim required for continuous buffer')
            self.action = np.zeros((capacity, action_dim), dtype=np.float32)

        self.size = 0
        self.head = 0

    def push(self, sample_idx, current_mask, action, reward, next_mask, done):
        i = self.head
        self.sample_idx[i]   = sample_idx
        self.current_mask[i] = current_mask
        self.next_mask[i]    = next_mask
        self.action[i]       = action
        self.reward[i]       = reward
        self.done[i]         = bool(done)
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            sample_idx   = self.sample_idx[idx],
            current_mask = self.current_mask[idx],
            next_mask    = self.next_mask[idx],
            action       = self.action[idx],
            reward       = self.reward[idx],
            done         = self.done[idx],
        )

    def __len__(self):
        return self.size
