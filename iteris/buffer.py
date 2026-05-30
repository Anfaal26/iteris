"""
Replay buffer for DRL agents.

Memory-optimised storage: instead of full (4, H, W) float32 states, we store
the sample_idx + binary mask (uint8) per transition. Static channels (image,
init_mask) are looked up at sample time.

For training-speed optimisation we ALSO cache the signed-distance transform
(SDT) per transition as float16. Without this cache, agent.update() would
recompute SDT 128× per gradient step via scipy.ndimage.distance_transform_edt
(CPU-bound, ~5 ms each) — about 80% of total training wall time.

Storage budget (10k transitions, 256×256 mask):
  current_mask + next_mask (uint8):   1.3 GB
  current_sdt  + next_sdt  (float16): 2.6 GB
  Total:                              ~4 GB  (vs. ~20 GB naive float32 states)
"""

from typing import Dict, Optional
import numpy as np


class ContourReplayBuffer:
    """Uniform circular replay buffer for the tracing paradigm.

    Unlike ``ReplayBuffer`` (which stores a compact sample_idx + mask and
    reconstructs the full refinement state at sample time), tracing states are
    small local patches that depend on the trajectory and cannot be rebuilt
    from a sample index. So the (4, patch, patch) patch tensors are stored
    directly. There is no SDT to cache here — the distance reward lives in the
    env, not the state.

    States are kept as float16 to halve host-RAM cost (patch values are image
    intensities in [0, 1] plus binary layers — fp16 is ample); they are
    upcast to float32 by the state-builder at update time.

    The ``sample()`` output reuses the key names ``current_mask`` / ``next_mask``
    so the existing ``DQNAgent.update(batch, state_builder)`` works unchanged:
    pair this buffer with a state-builder that simply upcasts the stored patch
    to a float32 tensor (see ``drl_training._make_patch_state_builder``).
    """

    def __init__(self, capacity: int, state_shape: tuple,
                 action_dim: Optional[int] = None, discrete: bool = True):
        if not discrete:
            raise ValueError('ContourReplayBuffer is discrete-only (8 directions)')
        self.capacity    = capacity
        self.state_shape = state_shape          # (4, patch, patch)
        self.cache_sdt   = False                # interface parity with ReplayBuffer

        self.current_state = np.zeros((capacity, *state_shape), dtype=np.float16)
        self.next_state    = np.zeros((capacity, *state_shape), dtype=np.float16)
        self.action        = np.zeros(capacity, dtype=np.int64)
        self.reward        = np.zeros(capacity, dtype=np.float32)
        self.done          = np.zeros(capacity, dtype=np.uint8)

        self.size = 0
        self.head = 0

    def push(self, current_state, action, reward, next_state, done):
        i = self.head
        self.current_state[i] = current_state.astype(np.float16)
        self.next_state[i]    = next_state.astype(np.float16)
        self.action[i]        = action
        self.reward[i]        = reward
        self.done[i]          = bool(done)
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return dict(
            sample_idx   = idx,                 # unused by the patch builder; kept for parity
            current_mask = self.current_state[idx],
            next_mask    = self.next_state[idx],
            action       = self.action[idx],
            reward       = self.reward[idx],
            done         = self.done[idx],
        )

    def __len__(self):
        return self.size


class ReplayBuffer:
    """Uniform circular replay buffer with cached SDT."""

    def __init__(
        self,
        capacity: int,
        mask_shape: tuple,
        action_dim: int = None,
        discrete: bool = True,
        cache_sdt: bool = True,
    ):
        self.capacity   = capacity
        self.mask_shape = mask_shape
        self.discrete   = discrete
        self.cache_sdt  = cache_sdt

        # Static reference per transition
        self.sample_idx   = np.zeros(capacity, dtype=np.int32)
        # Dynamic state at t and t+1
        self.current_mask = np.zeros((capacity, *mask_shape), dtype=np.uint8)
        self.next_mask    = np.zeros((capacity, *mask_shape), dtype=np.uint8)
        self.reward       = np.zeros(capacity, dtype=np.float32)
        self.done         = np.zeros(capacity, dtype=np.uint8)

        if cache_sdt:
            # float16 keeps SDT precision adequate (it's already clipped to ±1)
            self.current_sdt = np.zeros((capacity, *mask_shape), dtype=np.float16)
            self.next_sdt    = np.zeros((capacity, *mask_shape), dtype=np.float16)
        else:
            self.current_sdt = None
            self.next_sdt    = None

        if discrete:
            self.action = np.zeros(capacity, dtype=np.int64)
        else:
            if action_dim is None:
                raise ValueError('action_dim required for continuous buffer')
            self.action = np.zeros((capacity, action_dim), dtype=np.float32)

        self.size = 0
        self.head = 0

    def push(
        self,
        sample_idx,
        current_mask,
        action,
        reward,
        next_mask,
        done,
        current_sdt: Optional[np.ndarray] = None,
        next_sdt:    Optional[np.ndarray] = None,
    ):
        i = self.head
        self.sample_idx[i]   = sample_idx
        self.current_mask[i] = current_mask
        self.next_mask[i]    = next_mask
        self.action[i]       = action
        self.reward[i]       = reward
        self.done[i]         = bool(done)
        if self.cache_sdt and current_sdt is not None:
            self.current_sdt[i] = current_sdt.astype(np.float16)
            self.next_sdt[i]    = next_sdt.astype(np.float16)
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            sample_idx   = self.sample_idx[idx],
            current_mask = self.current_mask[idx],
            next_mask    = self.next_mask[idx],
            action       = self.action[idx],
            reward       = self.reward[idx],
            done         = self.done[idx],
        )
        if self.cache_sdt:
            batch['current_sdt'] = self.current_sdt[idx]
            batch['next_sdt']    = self.next_sdt[idx]
        return batch

    def __len__(self):
        return self.size
