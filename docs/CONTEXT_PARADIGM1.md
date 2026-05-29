# Paradigm 1: Sequential Boundary Tracing — Context & Design

*Authored 2026-05-29. Companion to PLAN_PARADIGM1.md.*

## 1. Why we are pivoting

The thesis topic is **DRL for Automated Medical Image Segmentation**. The system as built before this pivot ("DRL refines a U-Net mask via morphological actions") is more accurately described as **DRL refinement of a U-Net baseline**. That is a defensible sub-topic but it is not literally what the title claims.

Paradigm 1 ("DRL traces a boundary contour point by point") matches the title exactly: the agent itself produces the segmentation. The U-Net's role shrinks to providing a seed point, which is consistent with how the published DRL-segmentation literature on cardiac data actually works (Yin et al. 2021 "Left Ventricle Contouring" arXiv:2106.04127; Mayy 2021 "Edge-Sensitive LV Segmentation" Sensors 21(7):2375).

Concrete benefits of the pivot:

- **Title alignment** — committee members reading the title will see the system do what the title says.
- **Literature alignment** — direct precedent in the DRL-cardiac-segmentation literature; expected performance can be benchmarked against published numbers (Yin et al. report competitive Dice on ACDC / Sunnybrook).
- **Stronger paper story** — the existing refinement system can be kept as a published baseline comparison. No published work has compared refinement-DRL against tracing-DRL on the same datasets with the same RL family. That comparison becomes the paper's experimental contribution.
- **DDPG question resolves cleanly** — sequential tracing is inherently discrete (8 directions), so the awkward "continuous global morph" gap disappears. DDPG either gets dropped or matches the literature's contour-vertex-displacement form (MARL-MambaContour / SAC-style), as future work.

## 2. The two paradigms compared

| | Current (refinement) | Paradigm 1 (tracing) |
|---|---|---|
| **What the agent edits** | The whole mask | A single point's position |
| **Action per step** | Modifies ~30–100 px at once via morphology | Moves 1 pixel in one of 8 directions |
| **Episode length** | 5–20 steps | 100–400 steps (perimeter-dependent) |
| **State** | Full image + masks `(4, 256, 256)` | Local patch around current point `(4, 64, 64)` |
| **Reward** | `Dice_t − Dice_0` | `−distance(point_t, GT_boundary)` |
| **Termination** | Convergence / improvement / max_steps | Contour closes / off-image / max_length / explicit stop |
| **What "fails" looks like** | Mask gets globally worse | Trace goes off the boundary or fails to close |
| **U-Net's role** | Provides the starting mask | Provides the seed point + reference boundary |

## 3. Algorithm — sequential boundary tracing

### 3.1 State representation `(4, 64, 64)`

| Channel | Content |
|---|---|
| 0 | Image patch (intensity) centred on the current point |
| 1 | Position layer — 1 at the current point, 0 elsewhere |
| 2 | Visited layer — 1 along the trajectory so far, 0 elsewhere (capped to the patch) |
| 3 | U-Net init boundary — 1 on the init mask's edge pixels within the patch |

Cropping is done in image coordinates; the agent sees a *moving window* over the image, anchored at its current location. Padding (zeros) is applied at image edges.

### 3.2 Action space — 8 directions

```
0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
```

Single-pixel step in the chosen direction. **Skip neighbourhoods (8 dirs × 2 step sizes = 16 actions)** are a stretch goal for v2 if step-1 traces are too slow to converge — kept out of the v1 scope to keep the comparison clean.

### 3.3 Reward shaping

```
r_step       = -distance(point_t, GT_boundary)         # in pixels, clipped to [-d_max, 0]
r_boundary   = +bonus_on_boundary                       # small +ve when distance < ε
r_closure    = +closure_bonus                           # terminal: trace returns to start
r_offimage   = -offimage_penalty                        # terminal: trace exits image
r_too_long   = -length_penalty                          # terminal: max_length exceeded without closure
```

Closure bonus is the dominant terminal reward — agent learns to seek closure proactively. Distance-based per-step reward keeps the agent on the boundary throughout the trace.

### 3.4 Episode lifecycle

```
reset():
  init_mask  = U-Net warm-start mask                    # from warm_start.precompute_init_masks
  seed_point = topmost_boundary_pixel(largest_cc(init_mask))
  trajectory = [seed_point]
  state      = extract_patch(image, position=seed_point, history=[seed_point], init_mask)

step(action):
  direction       = DIRECTIONS[action]                  # 8 cardinal+diagonal unit vectors
  new_point       = current_point + direction
  trajectory.append(new_point)
  distance        = min_dist(new_point, GT_boundary_edge_pixels)
  r               = -distance + boundary_bonus(distance) + termination_rewards
  done            = closure_check() or off_image() or len(trajectory) >= max_length
  state           = extract_patch(image, new_point, trajectory, init_mask)
  return state, r, done, info

on done:
  final_mask = rasterise(trajectory)                     # skimage.draw.polygon → uint8 mask
  test_metrics = dice(final_mask, gt_mask), hd95(final_mask, gt_mask)
```

### 3.5 Seed point selection

**Default heuristic**: topmost boundary pixel of the largest connected component of the U-Net init mask. Deterministic, single line of code, no extra training required.

**Stretch goal** (v2): train a small First-P-Net (Yin et al. style) to predict the seed point directly from the image. Improves robustness on samples where U-Net init is wrong about location.

### 3.6 Closure detection

```
closed = distance(current_point, seed_point) < closure_tolerance
       AND len(trajectory) >= min_perimeter_steps
```

`min_perimeter_steps` prevents degenerate "immediately close" exploits (agent walks one step, returns home, claims closure). Default: 50 steps for CAMUS, 30 for BRISC.

## 4. File structure

### 4.1 What's new

```
iteris/
  env_contour.py            ContourTracingEnv class
  contour_utils.py          seed_point(), closure_check(), rasterise_trajectory()
docs/
  CONTEXT_PARADIGM1.md      this document
  PLAN_PARADIGM1.md         day-by-day implementation plan
```

### 4.2 What's extended

```
iteris/
  drl_networks.py           +PatchQNetwork, +PatchDuelingQNetwork (smaller input)
  buffer.py                 +ContourReplayBuffer (stores patch state tensors directly)
  drl_training.py           ENV_REGISTRY adds 'contour_tracing';
                            evaluate_agent rasterises trace before computing metrics
configs/
  brisc_drl_tumor.yaml      per-agent block: env_type (refinement | contour_tracing)
  camus_drl_c1/c2/c3.yaml   same
notebooks/
  03a/b/c, 04                add trace-replay viz section (point-by-point animation)
```

### 4.3 What's preserved unchanged

```
iteris/
  env.py                    refinement env stays as baseline comparison
  agents.py                 DQN/DDQN/Dueling/MSA-Dueling all work for both paradigms
  msa.py                    MSA backbone is input-size agnostic
  warm_start.py             U-Net inference still seeds init masks
  config.py                 config loading unchanged
```

The agents code never knows whether it's refining or tracing — both paradigms expose the same `(state, action, reward, done)` contract.

## 5. Per-dataset notes

### 5.1 CAMUS

- Single closed contour per class — natural fit
- Expected trace length: LV_endo ~250 steps, LV_epi ~320, LA ~280
- `max_trace_length: 1.5× expected_perimeter`
- `closure_tolerance: 3 px`

### 5.2 BRISC

- Tumours are blobs, not contours — still works (every blob has a boundary)
- Always uses **largest connected component** of GT mask for distance reward and U-Net init for seeding
- Multi-focal tumours (rare) — secondary component is ignored at the cost of imperfect Dice on those samples; documented as a v1 limitation
- Expected trace length: ~100 steps for typical 30-px tumour
- `closure_tolerance: 4 px` (looser because irregular tumour shapes)
- `min_perimeter_steps: 30` (smaller because tumours are smaller)

## 6. What this commits to (and what it does not)

### 6.1 In-scope for v1

- DQN, DDQN, Dueling DQN, MSA-Dueling DQN on `ContourTracingEnv`
- All 4 datasets/classes (CAMUS c1/c2/c3 + BRISC tumor)
- Heuristic seed point (no First-P-Net)
- 8-direction discrete action (no skip neighbourhoods)
- Single-largest-component handling on BRISC

### 6.2 Out-of-scope / future work

- First-P-Net for learned seed-point prediction
- Skip neighbourhoods (16-action variant)
- Contour DDPG (matching MARL-MambaContour) — the continuous analogue of this paradigm
- Multi-component tracing on BRISC
- 3D tracing (only 2D considered here)

## 7. What survives in the paper as the refinement-DRL baseline

The current 13-action / 9-action directional morphology refinement system stays in the codebase under `env.py`. Its training results on CAMUS c1/c2/c3 and BRISC become a **baseline column in the paper's experimental table**:

> Table N: DRL formulations on CAMUS and BRISC. Refinement-DRL takes the U-Net mask as input and applies morphological edits; Tracing-DRL discards the U-Net mask except for seeding the first point.

This makes the paper richer than either approach alone — it directly answers the under-explored question of "should DRL edit a mask or draw one?" on the same data with the same RL algorithm family.
