# Iteris — Project Context

> **Use this file as the seed context for UI design work, fresh LLM sessions, or onboarding collaborators.** Self-contained snapshot of scope, progress, and the UI brief.

---

## 1. What This Is

**Iteris** is a research-paper + UI dual deliverable from a 5-week Taylor's University capstone (PRJ63504).

The research builds and compares six **Deep Reinforcement Learning** agents that refine medical image segmentation masks produced by a U-Net baseline. The UI is a research demo that visualises how each agent iteratively improves a segmentation over up to 20 steps.

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
| **CAMUS** | Cardiac ultrasound | LV endo / LV epi / Left atrium | Primary training set, ~500 patients |
| **CHAOS** | CT + MRI | Liver / Kidneys / Spleen | Cross-modal transfer eval (Week 4) |
| ACDC, BraTS, ISIC, DRIVE | Various | Various | Stretch — transfer/few-shot only |

**Baseline model:** Attention Residual U-Net (ResNet-34 encoder, attention gates, transposed-conv decoder).

**DRL agents (6 total)**
| Agent | Action space | Key detail |
|---|---|---|
| DQN | Discrete (7) | CNN policy, replay 10k, ε 1.0→0.05 |
| DDQN | Discrete (7) | Decoupled action selection / evaluation |
| Dueling DQN | Discrete (7) | V(s) + A(s,a) split head |
| DDPG | Continuous (2D) | OU noise, τ=0.005 |
| MSA-Dueling DQN | Discrete + attention | 4-head MSA, 64-dim keys |
| MSA-DDPG | Continuous + attention | 4-head MSA in actor |

**RL Environment (locked v2 contract)**
- **State:** `(4, 256, 256)` float32
  - ch 0: image (normalised [0,1])
  - ch 1: current binary mask for the target class
  - ch 2: signed distance transform of current mask, normalised ±1
  - ch 3: **U-Net init mask** (fixed throughout episode — gives agent its "starting point" reference)
- **Discrete actions (7):** dilate · erode · shift ↑ · shift ↓ · shift ← · shift → · no-op
- **Continuous actions (2):** (dy, dx) ∈ [-0.1, 0.1] (fractional translation)
- **Reward:** `r_t = Dice(mask_t, GT) - Dice(mask_{t-1}, GT)` (first-order delta — telescopes to total Δ Dice)
- **Episode end:** step ≥ 20, OR composite early-stop: |ΔDice| < 0.001 AND |ΔHD95| < 0.5 px for 3 consecutive steps
- **Formulation:** per-class binary agents (one episode per structure). Final multi-class mask = union of per-class refinements.

**Evaluation metrics:** Dice · IoU · HD · HD95 · Wilcoxon signed-rank tests (p<0.05) · 5-fold CV.

---

## 3. Current Status (live)

**Week 1 — DONE** ✓
- Attention Residual U-Net trained on CAMUS
- Achieved test results:
  - LV endo Dice **0.9378** (target was ≥ 0.85)
  - LV epi Dice **0.8723**
  - LA Dice **0.8958**
  - Mean Dice **0.9020**, mean HD95 ~6 px (~2.3 mm)
- Outputs available: `camus_best.pt` checkpoint, per-patient scores CSV, predicted masks per test sample, learning curves PNG.

**Week 2 — IN PROGRESS**
- `SegmentationEnv` v2 written + validated (state, reward, action set all locked)
- Compute benchmark complete on T4
- Next: random-action baseline, then DQN/DDQN/Dueling agents

**Weeks 3–5 — PLANNED**
- W3: DDPG + MSA variants on CAMUS
- W4: Unified eval harness · ablations · CHAOS cross-modal transfer · stats
- W5: Paper draft + Iteris UI deployment

**File outputs the UI will consume**
- `camus_best.pt` — U-Net baseline checkpoint (loaded once at backend startup)
- `<agent>_camus.pt` — six DRL agent checkpoints (loaded once each)
- `camus_test_scores.csv` — per-patient metrics for U-Net (extend per agent in Week 2+)
- `camus_pred_masks/*.npy` — U-Net predicted masks per test sample (used as RL env init state)
- `<agent>_episode_traces/*.npy` — per-step masks for 20-step playback (generated Week 3)

---

## 4. The UI Brief

**Backend:** FastAPI + PyTorch. Loads 1 U-Net + 6 DRL agents at startup.
**Frontend:** HTML/CSS/JS or React. Canvas-based image viewer.
**Hosting target:** Hugging Face Spaces or Render (free tier).

**Endpoints**
- `POST /predict` — upload image, return baseline + all-agent predictions
- `GET /models` — list available models with metadata (name, family, val Dice, training dataset)
- `POST /compare` — return side-by-side or wipe-comparison data for selected models
- `GET /trace/{agent}/{sample_id}` — return the 20 per-step masks for iteration playback

**Pages**
| Page | Purpose |
|---|---|
| **Workspace** | Upload + model select (left) · canvas viewer (centre) · results panel (right) |
| **Wipe Comparison** | Draggable divider: U-Net vs DRL agent, shared zoom/pan |
| **Iteration Playback** | Step scrubber 0–20, transport controls, reward sparkline, per-step annotations |
| **Side-by-Side** | 3 model columns, shared zoom/pan, dynamic "Best Dice" badge, metric bar chart |
| **Model Library** | All 7 models, expandable training curves, filter by family/metric/dataset |
| **Dataset Explorer** | CAMUS + CHAOS grid, difficulty tags, curated examples including 1 deliberate failure case |

**Key components**
- **Preprocessing status:** step indicator with elapsed ms per stage (no spinner)
- **Model cards:** Dice / IoU / HD / HD95 pills; active card has teal left border
- **Results panel:** metric cards with green/amber/red benchmark coding · per-structure table
- **W/L controls:** Window / Level sliders + modality presets (CT / MRI / US) — **non-negotiable**
- **Viewer:** scroll-zoom up to 8×, drag-pan, minimap, pixel coord + intensity readout on hover
- **Export:** PNG mask · JSON metrics · shareable session link
- **Demo mode:** 5–6 preloaded test samples for zero-upload demos

**Accessibility:** WCAG 2.1 AA · full keyboard navigation · colour-blind pattern overlays · Reduce Motion mode.

---

## 5. Design System (locked)

**Theme**
- Light default: `#F7F8FA` background · `#FFFFFF` surface
- "Reading Room" dark mode toggle
- Accent: `#0A7EA4` teal
- Text: `#0F1923` primary · `#64748B` secondary · `#E2E8F0` borders

**Semantic colours**
- Success `#16A34A` · Warning `#D97706` · Error `#DC2626` · Uncertainty `#7C3AED`

**Structure mask colours** (used ONLY on masks — never reused in UI chrome)
- LV endo `#00C9A7` · LV epi `#F59E0B` · LA `#F87171`
- Liver `#818CF8` · Kidneys `#34D399` · Spleen `#FB923C`
- (Expansion palette — RV `#60A5FA` · Myocardium `#A78BFA` · Brain tumour core `#F43F5E` · Retinal vessel `#22D3EE` · Skin lesion `#FBBF24` · Liver tumour `#EF4444` · Pancreas `#8B5CF6`)

**Typography**
- Sora (headings) · DM Sans (body) · JetBrains Mono (metrics)

**Motion**
- **Zero animation on clinical output (masks, predictions)**
- 100–200 ms ease-out on UI chrome only

---

## 6. Tools & Stack

- **Training:** PyTorch · MONAI (transforms, losses, metrics) · scipy.ndimage (env-side mask ops)
- **Compute:** Kaggle T4 GPU (P100 dropped — current PyTorch lacks Pascal kernels)
- **Tracking:** W&B for live training metrics · CSV exports for paper figures
- **Repo:** github.com/Anfaal26/iteris (private)
- **Versioning:** All hyperparameters in YAML configs · no hardcoded values · fixed seeds (PyTorch + NumPy + Python random)

---

## 7. Non-negotiables for the UI

1. **Never present output as clinical.** The product is a research demo. Every page header must communicate this.
2. **Window/Level controls are required** for medical imaging credibility.
3. **Iteration playback is the differentiator.** The DRL→U-Net step-by-step view is what makes this project novel; it must be the most polished page.
4. **Failure cases must be included** in Dataset Explorer. Hiding them undermines the research framing.
5. **Mask colours are sacred** — they encode anatomical class. Do not reuse them in buttons, accents, or chrome.

---

## 8. Risks the UI Should Defuse

| Risk | UI mitigation |
|---|---|
| User mistakes demo output for clinical use | Persistent "Research demo · not for clinical use" banner |
| Agents fail catastrophically on out-of-distribution input | Show uncertainty pills · gate Export behind a "low confidence" warning |
| Iteration playback feels like animation, not signal | Step scrubber (not auto-play) · reward sparkline shows what each step "earned" |
| User can't tell which agent is best | "Best Dice" badge dynamically updates · benchmark colour-coding on metric cards |

---

*Last updated when committed. For the current Week 1 results, see `camus_best_summary.json` in the latest Kaggle output.*
