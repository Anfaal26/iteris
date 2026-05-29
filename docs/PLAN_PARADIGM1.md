# Paradigm 1 Implementation Plan — Sequential Boundary Tracing

*Companion to CONTEXT_PARADIGM1.md. Authored 2026-05-29. Target completion: 2026-06-04.*

This document is the day-by-day implementation contract. Each day produces a concrete deliverable that's individually testable. If any day slips, the rollback section at the end describes how to stop without breaking the existing refinement system.

## Overview

| Day | Theme | Deliverable | Validation |
|---|---|---|---|
| 1 | Core env + utils | `ContourTracingEnv` + `contour_utils.py` | Unit-level smoke test: reset → 10 steps → done |
| 2 | Reward + termination tuning | Distance reward, closure detection working end-to-end | Trace closes successfully on ≥ 1 synthetic sample |
| 3 | Networks + buffer | `PatchQNetwork`, `PatchDuelingQNetwork`, `ContourReplayBuffer` | 100-step training cycle without errors |
| 4 | Training loop + agent wiring | `drl_training.py` extended; env-type dispatch via config | End-to-end dry run on CAMUS c1 (600 steps) |
| 5 | Notebooks + visualisation | Trace-replay viz, rasterise-to-mask eval, per-class configs | All 4 notebooks parse and run a 600-step dry run |
| 6 | Debug + first real run | Full-length CAMUS c1 DDQN training run | Best val Dice ≥ 0.50 (proof of life; full performance comes later) |

Buffer: 2 days for inevitable contour-closure edge cases that will only surface once a real agent starts exploring.

## Day 1 — Core env + utils

### Files created

`iteris/contour_utils.py`:

```python
def seed_point_from_init_mask(init_mask, method='topmost'):
    """Returns (y, x) of seed point on boundary of largest CC."""

def is_closed(trajectory, seed_point, closure_tolerance, min_steps):
    """Returns True if trajectory has returned to seed within tolerance."""

def is_off_image(point, H, W):
    """Returns True if point is outside image bounds."""

def rasterise_trajectory(trajectory, H, W):
    """Convert trajectory (list of (y, x)) to binary mask via polygon fill."""

def boundary_edge_pixels(mask):
    """Returns (N, 2) array of (y, x) coordinates of mask boundary."""

def distance_to_boundary(point, boundary_pixels):
    """Euclidean distance from point to nearest boundary pixel."""

# 8-direction action lookup
DIRECTIONS = np.array([
    [-1,  0],  # 0: N
    [-1, +1],  # 1: NE
    [ 0, +1],  # 2: E
    [+1, +1],  # 3: SE
    [+1,  0],  # 4: S
    [+1, -1],  # 5: SW
    [ 0, -1],  # 6: W
    [-1, -1],  # 7: NW
], dtype=np.int32)
```

`iteris/env_contour.py`:

```python
class ContourTracingEnv:
    NUM_DISCRETE_ACTIONS = 8

    def __init__(
        self, image, gt_mask, init_mask,
        patch_size=64, max_trace_length=400,
        closure_tolerance=3, min_perimeter_steps=50,
        boundary_bonus_distance=1.5,
        reward_offimage=-10.0, reward_closure=+5.0, reward_length_penalty=-5.0,
        max_distance_penalty=10.0,
        seed_method='topmost',
    ): ...

    def reset(self) -> np.ndarray: ...
    def step(self, action) -> (state, reward, done, info): ...
    def _extract_patch(self) -> np.ndarray:
        """Returns (4, patch_size, patch_size): image, position, visited, init-edge."""
    def get_final_mask(self) -> np.ndarray:
        """Rasterise trajectory to binary mask. Called at episode end."""
```

### Smoke test (end of Day 1)

```python
env = ContourTracingEnv(image, gt_mask, init_mask)
state = env.reset()
assert state.shape == (4, 64, 64)
for _ in range(10):
    state, r, done, info = env.step(np.random.randint(8))
    assert state.shape == (4, 64, 64)
print('Day 1 smoke test passed')
```

## Day 2 — Reward + termination tuning

### Synthetic sample test

Build a 256² image with a known circular tumour at centre, radius 50 px. Manually scripted "perfect trace" walks around the circle. Verify:

- Reward sums to large positive value (closure bonus + per-step boundary bonuses)
- Episode terminates with `done=True` and `info['closed']=True`
- Rasterised mask has Dice ≥ 0.95 against the ground-truth circle

### Failure-mode tests

- Random walker → terminates on off-image or max_length, large negative cumulative reward
- Tiny trace (5 steps then return to seed) → terminates on `min_perimeter_steps` not met, no closure bonus

### Deliverable

`tests/test_contour_env.py` with three named test cases that all pass.

## Day 3 — Networks + buffer

### Files modified

`iteris/drl_networks.py` — add patch-input variants:

```python
class PatchQNetwork(nn.Module):
    def __init__(self, in_channels=4, num_actions=8, patch_size=64, embed_dim=128): ...

class PatchDuelingQNetwork(nn.Module):
    def __init__(self, in_channels=4, num_actions=8, patch_size=64, embed_dim=128): ...
```

Smaller embed_dim (128 vs 256) because the input is 4× smaller spatially and the action space is much smaller.

`iteris/msa.py` — `MSABackbone` is already input-size agnostic (uses adaptive pooling). Verify it works at 64×64 input; possibly adjust the spatial token grid from 8×8 to 4×4.

`iteris/buffer.py` — add `ContourReplayBuffer`:

```python
class ContourReplayBuffer:
    """Replay buffer for ContourTracingEnv.

    Stores patch state tensors directly. No SDT caching (irrelevant for tracing).
    """
    def __init__(self, capacity, state_shape, action_dim=None, discrete=True):
        self.state_shape = state_shape   # (4, 64, 64)
        ...
```

### Validation

A standalone DDQN agent created with PatchQNetwork should accept a 100-batch update without shape errors:

```python
agent = DDQNAgent(num_actions=8)
# manually swap network for patch variant
agent.q = PatchQNetwork(num_actions=8)
agent.q_target = deepcopy(agent.q)
# push 100 transitions to buffer, sample batch, call agent.update
```

## Day 4 — Training loop + agent wiring

### Files modified

`iteris/drl_training.py` — extend `ENV_REGISTRY`:

```python
from .env_contour import ContourTracingEnv

ENV_REGISTRY = {
    'default':            SegmentationEnv,
    'brisc_small_target': SegmentationEnvBRISC,
    'contour_tracing':    ContourTracingEnv,
}
```

Add dispatch logic for state shape: refinement envs use `(4, H, W)` for the full image; tracing env uses `(4, patch_size, patch_size)`. Network and buffer pick the right size from the env's `state_shape` attribute.

`evaluate_agent` for tracing: run episode, get `env.get_final_mask()`, compute Dice/HD95 vs GT.

### Per-class config additions

`configs/camus_drl_c1.yaml` (example):

```yaml
agents:
  DDQN_TRACE:
    agent_type:           DDQN
    env_class:            contour_tracing
    patch_size:           64
    max_trace_length:     400
    closure_tolerance:    3
    min_perimeter_steps:  50
    boundary_bonus_distance: 1.5
    reward_closure:       5.0
    reward_offimage:     -10.0
    reward_length_penalty: -5.0
    # rest as before: lr, gamma, tau, embed_dim, etc.
```

Per-agent block can now declare `env_class`, so `DDQN` (refinement) and `DDQN_TRACE` (tracing) coexist in the same config file. The notebook's `AGENT_NAME` selector picks one.

### Dry-run validation (end of Day 4)

`AGENT_NAME='DDQN_TRACE'` in `03a_camus_drl_lv_endo.ipynb`, run §3 with 600 train_steps:

```
[drl] env_class=contour_tracing (ContourTracingEnv) | discrete actions=8 | action_type=discrete
[drl] Pre-filling buffer with 100 random transitions...
[drl] Training: 600 steps  →  /kaggle/working/camus_ddqn_trace_c1_best.pt
...
✓ Dry run passed.
```

## Day 5 — Notebooks + visualisation

### Notebook updates (4 files)

`03a/b/c_camus_drl_*` and `04_brisc_drl`:

- §5 (vizsetup): dispatch on `cfg.get('env_class', 'default')` — if tracing, set `IS_TRACING = True` and import contour-aware viz helpers.
- §6 (sample comparison): for tracing, show U-Net init mask + final rasterised trace + GT. Skip the per-step rolling-mask viz used for refinement (replaced by §7).
- §7 (playback): for tracing, plot the trajectory point-by-point on the image (matplotlib `.plot()` with a colormap on `t`).
- §8 (behaviour analysis): for tracing, plot direction-distribution histogram (which of the 8 directions is preferred) instead of action-histogram. Trajectory length distribution as a second plot.
- §10 (test eval): for tracing, rasterise each test trajectory before computing Dice / HD95. Same JSON output format.

### Validation

All 4 notebooks open cleanly in Jupyter locally, kernel runs §0 → §5 without errors. Smoke test for both `AGENT_NAME='DDQN'` (refinement, existing) and `AGENT_NAME='DDQN_TRACE'` (tracing, new).

## Day 6 — Debug + first real run

### Real-data training

`03a_camus_drl_lv_endo.ipynb` on Kaggle T4 with `AGENT_NAME='DDQN_TRACE'`, full training run (~3–4 hr expected, since episodes are 200–400 steps × 30000 train_steps × ~5 steps/sec).

### Success criteria (proof of life)

- Training completes without errors
- Val Dice ≥ 0.50 by end of training (this is *proof of life*, not final performance — full tuning comes after the architecture is verified to work)
- Trace traces visibly follow the LV boundary in §7 playback
- Closure rate ≥ 60% (at least 60% of episodes successfully close into a polygon)

### Known risks to debug

| Risk | Diagnostic | Fix |
|---|---|---|
| Trace zigzags but never closes | Closure rate < 30% in info logs | Increase `reward_closure`, decrease `closure_tolerance` |
| Trace walks straight off image | Offimage rate > 20% | Increase `reward_offimage` (more negative) |
| Trace stays at seed (no exploration) | Trajectory length distribution dominated by ≤ 10 | Increase `min_perimeter_steps`; warmstart with random walk |
| Q-values explode | Training loss diverges | Lower `lr` to 5e-5; tighter `reward_clip` |
| Rasterised polygon has bad Dice despite good trace | Dice gap > 0.2 between trace-on-boundary frequency and final Dice | Inspect `rasterise_trajectory()` — likely a polygon-fill orientation issue |

## Rollback plan

If by **end of Day 4** the dry run does not run end-to-end:

- The new files (`env_contour.py`, `contour_utils.py`, patch network variants, contour buffer) are additions, not modifications to existing code. The existing refinement system is untouched.
- Revert: `git revert` the contour-related commits, or simply do not select `*_TRACE` agent names. The existing `DDQN` agent on the refinement env continues to work.
- Decision point: continue debugging on Day 5–6, or shelf the pivot and ship the refinement system as the only paradigm.

If **after Day 6**, traces don't close reliably (closure rate < 30%):

- This indicates a fundamental reward-shaping issue, not a code bug.
- Mitigation 1: switch to 16-action skip neighbourhood (faster perimeter coverage).
- Mitigation 2: add a learned First-P-Net for better seed points.
- Mitigation 3 (last resort): keep refinement as the only paradigm in v1 of the paper; document tracing pivot as future work.

## Out-of-scope explicitly

These are not part of v1 and should not be touched during the 6 days:

- Continuous DDPG on contour vertices (MARL-MambaContour style)
- 3D extension
- Skip neighbourhoods (8 × 2 = 16 actions)
- Learned seed-point predictor (First-P-Net)
- Multi-component handling on BRISC
- Mid-trace lookahead / planning agents (e.g., model-based RL)

## Definition of done (full plan)

- All 4 active notebooks support `AGENT_NAME` ending in `_TRACE` for tracing-paradigm agents
- `git log` shows 6 commits, one per day, each individually reverting cleanly
- Real training run on CAMUS c1 with DDQN_TRACE gives val Dice ≥ 0.60 and closure rate ≥ 60%
- `CONTEXT.md` (top-level) updated to mention both paradigms
- `docs/CONTEXT_PARADIGM1.md` and `docs/PLAN_PARADIGM1.md` updated with any deviations from the original plan

After Day 6: extend to the other 3 datasets/classes (CAMUS c2, c3, BRISC) by changing config knobs only — no code changes expected.
