# Iteris — Project Context

> **Use this file as the seed context for UI design work, fresh LLM sessions, or onboarding collaborators.** Self-contained snapshot of scope, progress, and the UI brief.

---

## 1. What This Is

**Iteris** is a research-paper + UI dual deliverable from a 5-week Taylor's University capstone (PRJ63504).

The research builds and compares Deep Reinforcement Learning agents that refine medical image segmentation masks produced by a U-Net baseline. The UI is a research demo that visualises how each agent iteratively improves a segmentation over up to 20 steps.

**Identity**

- Name: **Iteris**
- Tagline: *"See how AI learns to see."*
- Type: **Research demo workstation — not a clinical tool**
- Differentiator: **DRL iteration playback** — watch an agent refine a boundary step by step

**Target paper venues:** IEEE JBHI (primary) · MICCAI 2026 workshop · MedIA (prestige)

---

## 2. The Research Pipeline

```
Raw scan ──► U-Net baseline mask ──► DRL agent refines for ≤20 steps ──► Final mask
   (image)        (init state)          (each step adjusts the boundary)
```

**Datasets**
| Dataset | Modality | Structures | Role |
|---|---|---|---|
| **CAMUS** | Cardiac ultrasound | LV endo / LV epi / Left atrium | Primary cardiac set, ~500 patients, 3-class |
| **BRISC** | Brain MRI | Tumour (binary) | Secondary set, 9 586 image pairs, single-class |

**Baseline model:** Attention Residual U-Net (ResNet-34 encoder, attention gates, transposed-conv decoder).

**DRL agents**
| Agent | Action space | Key detail |
|---|---|---|
| DQN | Discrete (7) | Baseline Q-learner, CNN Q-head, ε-greedy, replay buffer |
| Dueling DQN | Discrete (7) | V(s) + A(s,a) split head — better at no-op credit assignment |
| DDPG | Continuous (3D) | OU noise, morph + dy + dx, τ=0.005 soft target update |
| PPO | Discrete (7) | On-policy clipped surrogate — planned; requires rollout buffer |
| MSA-\[Best\] | Same as base | 4-head multi-scale attention on top of best discrete/continuous agent |

> **MSA selection:** After Dueling, DDPG, and PPO training is complete, the top-performing agent type gets an MSA variant (MSA-Dueling or MSA-DDPG). Both MSA variants are pre-implemented and ready to run.

**RL Environment (locked v2 contract)**
- **State:** `(4, 256, 256)` float32
  - ch 0: image (normalised [0,1])
  - ch 1: current binary mask for the target class
  - ch 2: signed distance transform of current mask, normalised ±1
  - ch 3: U-Net init mask (fixed throughout episode — gives agent its "starting point" reference)
- **Discrete actions (7):** dilate · erode · shift ↑ · shift ↓ · shift ← · shift → · no-op
- **Continuous actions (3):** (morph, dy\_norm, dx\_norm) — morphological shift on SDT + fractional translation
- **Reward (episode-start baseline):** `r_t = metric(t) − metric(0)` — no step-wise oscillation trap
- **Reward modes:** `dice_hd_composite` (cardiac, boundary precision) · `iou_delta` (BRISC, small targets)
- **Episode end:** step ≥ 20, OR convergence: |ΔDice| < stop\_eps for stop\_n consecutive steps, OR improvement-maintained: all last stop\_n Dice > Dice\_0 + stop\_eps
- **Formulation:** per-class binary agents; final multi-class mask = union of per-class refinements

**Evaluation metrics:** Dice · IoU · HD · HD95 · Wilcoxon signed-rank tests (p<0.05) · 5-fold CV.

---

## 3. Current Status (live)

**Week 1 — DONE** ✓

*CAMUS Baseline (Attention Residual U-Net — cardiac ultrasound)*
| Structure | Dice | HD95 |
|---|---|---|
| LV endo | **0.9378** | ~6 px |
| LV epi | **0.8723** | — |
| Left atrium | **0.8958** | — |
| **Mean** | **0.9020** | ~6 px (~2.3 mm) |

Checkpoint: `camus_best.pt` · val Dice 0.9020 · target was ≥ 0.85 ✓

*BRISC Baseline (Attention Residual U-Net — brain tumour MRI)*
| Structure | Dice (val) | Dice (test) | HD95 (test) |
|---|---|---|---|
| Tumour | 0.8290 | **0.8351** | 8.36 px |

Checkpoint: `brisc_best.pt` · trained 60 epochs · target was ≥ 0.80 ✓  
Dataset: 9 586 pairs, images resized 512→256, z-score normalised, variable aspect ratios.  
Notable: per-patient HD95 varies widely (1.4 px → 78 px) — boundary refinement has strong upside.

**Week 2 — IN PROGRESS**

DRL environment v2 locked and validated. Reward architecture overhauled:
- Episode-start baseline reward (eliminates step-wise oscillation trap)
- Improvement-maintained early stopping (terminates dilate/erode 2-cycles)
- Largest-CC filtering on HD95 (removes stray U-Net FP pixels inflating to 200+ px)
- Per-class YAML configs for CAMUS (3 classes) and BRISC (1 class)

Current: DDQN running on CAMUS LV\_endo. Remaining agents queued.

Agents to implement (in order): DQN · DDQN · Dueling · DDPG · (PPO) · MSA-\[best\]  
Platforms: Kaggle T4 GPU · notebooks `06a/b/c` (CAMUS) · notebook `07` (BRISC)

**Week 3 — PLANNED: Full training runs**

- All 5–6 agents × 2 datasets × per-class (CAMUS: 3 configs, BRISC: 1 config)
- Checkpoint each best val Dice agent per run
- Generate per-step episode traces for UI playback (20-step `.npy` stacks)
- Ablation: reward mode · buffer size · epsilon schedule

**Week 4 — PLANNED: Evaluation + stats**

- Unified eval harness over all checkpoints
- Wilcoxon signed-rank vs U-Net baseline (per-class)
- 5-fold cross-validation
- Failure case curation for Dataset Explorer

**Week 5 — PLANNED: Paper + UI deployment**

- Paper draft targeting IEEE JBHI
- Iteris UI deployed on Hugging Face Spaces or Render

**File outputs the UI will consume**
- `camus_best.pt` / `brisc_best.pt` — U-Net baseline checkpoints
- `<agent>_camus_c<n>_best.pt` / `<agent>_brisc_tumor_best.pt` — DRL checkpoints
- `*_test_results.json` — per-agent test metrics
- `<agent>_episode_traces/*.npy` — per-step masks for playback (generated Week 3)

---

## 4. The UI Brief

**Backend:** FastAPI + PyTorch. Loads U-Net + DRL agents at startup.  
**Frontend:** React + Three.js (landing), Canvas-based viewer (workspace).  
**Hosting target:** Hugging Face Spaces or Render (free tier).

**Endpoints**
- `POST /predict` — upload image, return baseline + all-agent predictions
- `GET /models` — list available models with metadata (name, family, val Dice, dataset)
- `POST /compare` — return wipe-comparison data for selected models
- `GET /trace/{agent}/{sample_id}` — return per-step masks for iteration playback

**Pages / Sections**
| Page | Purpose |
|---|---|
| **Landing** | Dark hero with 3D element + 2 CTAs → Research hub or Workspace |
| **Research Hub** | Paper stats, methodology, training curves, ablation results |
| **Workspace** | Upload + model select (left) · canvas viewer (centre) · results panel (right) |
| **Wipe Comparison** | Draggable divider: U-Net vs DRL agent, shared zoom/pan |
| **Iteration Playback** | Step scrubber 0–20, transport controls, reward sparkline, per-step annotations |
| **Side-by-Side** | 3 model columns, shared zoom/pan, dynamic "Best Dice" badge, metric bar chart |
| **Model Library** | All agents, expandable training curves, filter by family/metric/dataset |
| **Dataset Explorer** | CAMUS + BRISC grid, difficulty tags, 1 deliberate failure case per dataset |

**Key components**
- **Preprocessing status:** step indicator with elapsed ms per stage (no spinner)
- **Model cards:** Dice / IoU / HD / HD95 pills; active card has accent left border
- **Results panel:** metric cards with green/amber/red benchmark coding · per-structure table
- **W/L controls:** Window / Level sliders + modality presets (MRI / US) — non-negotiable
- **Viewer:** scroll-zoom up to 8×, drag-pan, minimap, pixel coord + intensity readout on hover
- **Export:** PNG mask · JSON metrics · shareable session link
- **Demo mode:** 5–6 preloaded test samples (CAMUS cardiac + BRISC tumour) for zero-upload demos

**Accessibility:** WCAG 2.1 AA · full keyboard navigation · colour-blind pattern overlays · Reduce Motion mode.

---

## 5. Design System

### Theme — Dark Primary

Iteris is a research workstation. The dark theme foregrounds the medical images (greyscale scans read best against near-black) and gives the product a distinctive, credible aesthetic.

| Token | Value | Usage |
|---|---|---|
| `surface-base` | `#0B0F1A` | Page background — deep navy, not pure black |
| `surface-card` | `#131929` | Cards, panels, modals |
| `surface-raised` | `#1C2438` | Hover states, active rows, dropdowns |
| `border` | `#2A3550` | Dividers, card outlines |
| `text-primary` | `#E8EDF5` | Headlines, values |
| `text-secondary` | `#8A9BB8` | Labels, captions, metadata |
| `accent` | `#3B82F6` | Primary interactive — blue replaces old teal in dark context |
| `accent-glow` | `rgba(59,130,246,0.15)` | Button halos, focus rings |

**Semantic colours**
- Success `#22C55E` · Warning `#F59E0B` · Error `#EF4444` · Uncertainty `#A78BFA`

**Structure mask colours** (sacred — used ONLY on masks, never reused in UI chrome)
- LV endo `#00C9A7` · LV epi `#F59E0B` · LA `#F87171`
- Brain tumour `#F43F5E`
- (Expansion: RV `#60A5FA` · Myocardium `#A78BFA` · Retinal vessel `#22D3EE` · Skin lesion `#FBBF24`)

**Typography**
- Sora (headings, landing hero) · DM Sans (body, UI) · JetBrains Mono (metrics, JSON output)

**Motion**
- **Zero animation on clinical output** (masks, predictions, metric readouts)
- Landing hero: 3D element rotates at 0.3 rpm, subtle particle drift (pauses on Reduce Motion)
- UI chrome: 150 ms ease-out only

---

### Landing Page — Design Specification

The landing page is the only page with expressive design. Everything beyond it is a workstation.

**Hero section (100 vh, dark `#0B0F1A`)**

```
┌─────────────────────────────────────────────────────────────────┐
│  iteris                                         [Research] [Try] │  ← minimal nav
│                                                                   │
│          ╔══════════════════════════════╗                        │
│          ║   [  3D ELEMENT — right ¾  ] ║                        │
│  See how ║   Rotating 3D brain/heart   ║                        │
│  AI       ║   mesh or abstract neural   ║                        │
│  learns  ║   particle cloud in         ║                        │
│  to see. ║   Three.js / WebGL          ║                        │
│          ╚══════════════════════════════╝                        │
│                                                                   │
│   [ Try Iteris → ]   [ Explore Research ↓ ]                     │
│    (accent fill)       (ghost, outline)                          │
│                                                                   │
│   ↓ scroll                                                       │
└─────────────────────────────────────────────────────────────────┘
```

3D element options (choose one):
- **Option A (preferred):** Three.js particle cloud that morphs between a rough tumour boundary and a refined one — literally shows the DRL refinement concept.
- **Option B:** Rotating stylised brain MRI cross-section with glowing contour lines animated in steps (0→20).
- **Option C:** Abstract neural attention map — coloured blobs shifting and focusing, suggestive of the MSA mechanism.

**Scroll sections (below hero, full-width, `#0B0F1A` bg)**

1. **The Problem** — Two-column: left text ("U-Net boundaries drift. Pixels get missed."), right: animated before/after mask wipe (U-Net vs GT, no agent — sets up the problem).
2. **Our Approach** — Animated pipeline diagram: `scan → U-Net → DRL agent (steps 1–N) → refined mask`. Each stage lights up in sequence on scroll entry.
3. **Results Snapshot** — Dark card grid with key numbers:
   - CAMUS Mean Dice **0.902** · BRISC Tumour Dice **0.835** (baseline) → DRL target improvement
   - HD95 reduction numbers once training completes
4. **How It Works** — 3 icon tiles: `1. Upload scan` · `2. Pick agent` · `3. Watch it learn`
5. **CTA footer** — Repeats both buttons. "Not a clinical tool. Research demo only."

**Branch destinations**
- **Try Iteris →** → `/workspace` (the segmentation workstation)
- **Explore Research ↓** → `/research` (paper abstract, methodology, result tables, training curves, model library)

---

## 6. Tools & Stack

- **Training:** PyTorch · MONAI (transforms, losses, metrics) · scipy.ndimage (env-side mask ops)
- **Compute:** Kaggle T4 GPU
- **Frontend (landing):** React + Three.js for 3D element
- **Frontend (workspace):** React + Canvas (no Three.js — pure 2D image viewer)
- **Tracking:** W&B for live training metrics · CSV exports for paper figures
- **Repo:** github.com/Anfaal26/iteris (private)
- **Versioning:** All hyperparameters in YAML configs · no hardcoded values · fixed seeds (42)

---

## 7. Non-negotiables for the UI

1. **Never present output as clinical.** The product is a research demo. Every page must communicate this.
2. **Window/Level controls are required** for medical imaging credibility.
3. **Iteration playback is the differentiator.** DRL step-by-step view is the most polished page.
4. **Failure cases must be included** in Dataset Explorer. Hiding them undermines research framing.
5. **Mask colours are sacred** — they encode anatomical/pathological class. Never reuse in buttons or chrome.
6. **The landing page 3D element must pause/disable** under `prefers-reduced-motion`.

---

## 8. Risks the UI Should Defuse

| Risk | UI mitigation |
|---|---|
| User mistakes demo output for clinical use | Persistent "Research demo · not for clinical use" banner |
| Agents fail catastrophically on out-of-distribution input | Uncertainty pills · gate Export behind a "low confidence" warning |
| Iteration playback feels like animation, not signal | Step scrubber (not auto-play) · reward sparkline shows what each step earned |
| User can't tell which agent is best | "Best Dice" badge dynamically updates · benchmark colour-coding on metric cards |
| 3D landing element distracts from tool | 3D stays on landing only; workspace is purely functional |

---

*Last updated 2026-05-25. CAMUS + BRISC Week 1 baselines complete. DRL training in progress (Week 2).*
