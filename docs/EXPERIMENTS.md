# Iteris — Experimental Protocol (Phases A / B / C)

> The data-regime experimental design: how much data each phase uses, why those
> sizes, and the methodology that keeps the comparisons honest. Companion to
> [CONTEXT.md](CONTEXT.md) §3. **Defined:** 2026-07-01.

---

## 0. The one-line question each phase answers

> *Given how much labelled data you actually have, is "train a lightweight U-Net
> + refine its contour with DRL" better than "train a U-Net directly"? And does
> the answer change as data gets scarcer / the agent's backbone gets richer?*

| Phase | Data budget | Agent | What it isolates |
|---|---|---|---|
| **A** | Full dataset | DuelingDDQN, TD3 (CNN backbone) | The baseline comparison when data is *not* the constraint |
| **B** | Low-data subset | DuelingDDQN, TD3 (CNN backbone) | Does DRL's advantage over supervised DL *grow* when labels are scarce? |
| **C** | Even smaller subset | best Phase-A/B agent with **MSA** backbone | Does richer (self-attention) spatial reasoning help most in the most data-starved regime? |

---

## 1. How much data — the numbers, and why

### 1.1 What the literature says about the "DRL-beats-DL" regime

The effect Phase B/C is built to demonstrate is documented, but it is **regime-specific** — it appears at *dozens-to-low-hundreds* of training images, not thousands:

- **PixelDRL-MG (Liu et al., 2025, *Scientific Reports*)** — the single most directly comparable paper (same datasets family: ACDC cardiac / BraTS brain; DQN/DuelingDQN among its baselines). It benchmarks explicitly at **50-shot and 100-shot**, and reports that *both* supervised-DL and DRL degrade vs. full data — the comparison *between* them in that regime is the contribution. This is the anchor for our two low-data levels.
- **Manis et al. BraTS series (arXiv 2008.02708 / 2010.10763 / 2012.13321)** — at **30 training images**, a DQN reached ~70% accuracy while an identically-architected supervised CNN reached ~11% (p ≈ 5.9×10⁻⁴³). The mechanism: the supervised CNN overfits before ~epoch 10 on 30 images; the RL agent learns a policy, not a label-memorisation. This anchors the *severe* end (Phase C).
- **General medical-segmentation reviews (2024–25)** converge on the same point: supervised CNNs "remain prone to overfit" on small (hundreds, even thousands) databases, and limited data is an unsolved challenge for *both* paradigms — so the honest framing is "relative advantage," not "DRL is immune."

**Take-away for sizing:** Phase B should sit around the **~100–200-image** ("100-shot") regime where supervised DL is functional-but-degrading; Phase C around the **~50–90-image** regime where it overfits hard and the DRL/warm-start robustness shows most.

### 1.2 The chosen levels

Implemented through the existing `label_frac` knob ([`iteris/splits.py`](../iteris/splits.py)), which is **patient-level and shrinks only the training pool** — val and test always stay at full size (see §2.1). `seed: 42` is fixed everywhere, so the subset is deterministic and identical between the lite-U-Net training and the DRL warm-start (no leakage — see §2.2).

| Dataset | Phase A | Phase B (`label_frac`) | Phase C (`label_frac`) |
|---|---|---|---|
| **CAMUS** (≈375 train patients ≈ 1500 train imgs, 4/patient) | `1.0` — 375 pts (~1500 img) | **`0.10`** — ~37 pts (**~148 img**) | **`0.05`** — ~18 pts (**~74 img**) |
| **BRISC** (≈3092 train imgs) | `1.0` — ~3092 img | **`0.05`** — (**~155 img**) | **`0.03`** — (**~93 img**) |

Why the per-dataset fractions differ but the **absolute image counts are deliberately comparable** (~150 for B, ~75–95 for C): reviewers reason in absolute labelled-image counts, and matching the two datasets to a common absolute budget makes the cross-dataset story ("the effect holds on both cardiac US and brain MRI at the same scarcity") clean. CAMUS needs a larger fraction because it has far fewer patients; its 4-images-per-patient correlation means patient-count is the honest unit anyway.

> The exact kept-patient / kept-image counts print at runtime (`[splits] Few-shot: keeping N/M train patients`) — record those actual numbers in the paper, not the nominal targets.

### 1.3 BRISC is *more* interesting in Phase B/C, not less

At full data, BRISC tumour has **negative** realistic headroom (lite baseline 0.836 ≥ attention competitor 0.835) — DRL had nothing to win. But that gap is expected to **invert under scarcity**: the attention U-Net (~33 M params) overfits hard on ~150 images, while the lite U-Net (~0.48 M) + DRL refinement is structurally more robust. So Phase B/C is precisely where BRISC DRL may become competitive — a result that *only* the low-data phases can surface. Treat BRISC Phase B/C as a genuine open question, not a foregone "no headroom."

---

## 2. Methodology — the rules that keep it honest

### 2.1 Shrink **train only**; keep val/test at full size

`patient_level_split` already enforces this — `label_frac` carves the *training pool* after val/test are set aside. Consequence: the **test set is identical across A/B/C**, so Dice numbers are directly comparable across phases (a clean learning curve), and the metric stays statistically meaningful even when training on 18 patients. **Never** shrink the test set to "match" the regime — that just makes the numbers noisy and incomparable.

### 2.2 Retrain the lite U-Net on the subset — **yes, required**

> *"Do I need to retrain the lite U-Nets?"* — **Yes, on the same subset.**

If you reuse the *full-data* lite U-Net as the DRL warm-start, the "low-data" claim is false: your warm-start secretly consumed all 1500 images, so nothing about the experiment mimics scarcity. The lite U-Net must be trained at `label_frac = B` (and `= C`). It's tiny (~0.48 M params), so this is cheap. The DRL warm-start then reads the *same* `label_frac` + `seed` from the baseline config ([`warm_start.py`](../iteris/warm_start.py) and [`training.py`](../iteris/training.py) both call `patient_level_split` with the baseline config's seed/frac), so agent and U-Net see the **identical** patient subset — leak-free by construction.

### 2.3 Retrain the attention U-Net competitor on the subset — **also yes**

> *"Should I use DRL on top of a new attention U-Net baseline also trained on low data?"*

Two distinct things, untangled:
- **DRL never refines the attention U-Net.** That's a settled project decision — the attention net is the fixed *competitor* (the "what supervised DL achieves with this budget" reference line); RL always refines the **lite** net. So "DRL on top of the attention net" is not the design and shouldn't be built.
- **But the attention competitor must itself be retrained at the same `label_frac`.** The comparison Phase B/C makes is: *given N images,* is it better to (a) train an attention U-Net on N, or (b) train a lite U-Net on N **+** refine with DRL? Comparing low-data DRL against a *full-data* attention net is not a fair fight and not the question. Both reference lines must see the same N.

### 2.4 The model set per dataset per low-data phase

For each of CAMUS and BRISC, at each of `label_frac ∈ {B, C}`:

1. **Lite U-Net @ low-data** — the RL warm-start *and* a reported baseline line.
2. **Attention U-Net @ low-data** — the supervised competitor line.
3. **DRL (DuelingDDQN + TD3) refining #1** — the proposed method.

**Headline comparison:** #3 vs #2 (and the gain #3 − #1). The hypothesis is that #3 − #1 (the DRL gain) is *larger* at B/C than at A, and that #3 may meet or beat #2 in the regime where the big supervised net overfits.

CAMUS note: one CAMUS subset (multi-structure labels) serves all three classes — so it's **one** lite U-Net + **one** attention U-Net per phase (each multiclass), feeding three DRL agents (c1/c2/c3) × two algorithms. Don't retrain a separate baseline per class.

---

## 3. How to actually run a phase (the mechanism)

No new code or duplicated notebooks. Phase = a `label_frac` value in the baseline config + a fixed seed:

1. **Set `label_frac`** in the baseline config(s) for the phase:
   - CAMUS: `configs/CAMUS/camus_lite.yaml` (lite) and `configs/CAMUS/camus.yaml` (attention competitor).
   - BRISC: `configs/BRISC/brisc_lite.yaml` and `configs/BRISC/brisc.yaml`.
   - Keep `seed: 42`.
2. **Retrain** lite + attention via the existing baseline notebooks (`notebooks/unet/01_camus_lite`, `03_camus_attnunet`, `02_brisc_lite`, `04_brisc_attnunet`). They read `label_frac` straight from config.
3. **Run DRL** via the existing per-class notebooks. They auto-detect the baseline checkpoint name through `utils.model_suffix`, which is **phase-aware** — a `label_frac < 1.0` appends `_lf<pct>` (e.g. `_lf10`) to every artifact name, so low-data checkpoints/summaries/masks never collide with the Phase-A ones, and the DRL warm-start auto-finds the right phased checkpoint with zero notebook edits.

**Artifact naming examples** (`utils.model_suffix`): Phase A lite → `camus_lite_unet_best.pt` (unchanged); Phase B lite → `camus_lite_unet_lf10_best.pt`; Phase B attention → `camus_lf10_best.pt`.

> Practical Kaggle note: because phased checkpoints have distinct names, you can keep all phases' checkpoints in the same output dataset without overwrites — but remember to re-upload the retrained `iteris-pkg`/checkpoint datasets and re-point the notebook's attached version each phase (the recurring staleness trap — see [SKILLS.md](SKILLS.md) §3).

---

## 4. Phase C — the MSA backbone (prerequisite work)

Phase C swaps the agent's CNN backbone for the archived **MSA (Multi-Head Self-Attention)** backbone (`MSABackbone`, `MSADuelingQNetwork` in [`iteris/archive/msa.py`](../iteris/archive/msa.py)) — self-attention over the 8×8 feature-map tokens instead of global-average-pool, giving explicit cross-position reasoning. Run on the single **best** Phase-A/B agent×class, at the Phase-C `label_frac` — **not** a full re-sweep of the agent×class matrix (the point is an architecture ablation on the strongest base, not another grid).

**This is not yet runnable** — MSA is archived, not wired in. Prerequisite (a focused, self-contained task, deliberately deferred from the data-regime work so it gets its own verification):
1. Un-archive `msa.py` → `iteris/msa.py`; restore `MSADuelingDQNAgent` from `iteris/archive/agents_legacy.py`.
2. Add an `'MSA-DUELING'` (and/or `'MSA-TD3'`) entry to `AGENT_REGISTRY` in `drl_training.py`.
3. Add a config block + notebook selector; confirm the 5-channel input + sector-action head wiring matches the current `ContourRefineEnv`.

---

## Sources

- Liu et al. (2025), *Pixel-level deep reinforcement learning for accurate and robust medical image segmentation*, Scientific Reports — [nature.com/articles/s41598-025-92117-2](https://www.nature.com/articles/s41598-025-92117-2)
- Unsupervised deep clustering + RL for MRI brain-tumour segmentation with very small training sets (BraTS, ~30 images) — [arXiv:2012.13321](https://arxiv.org/pdf/2012.13321)
- A Systematic Review of Few-Shot Learning in Medical Imaging — [arXiv:2309.11433](https://arxiv.org/pdf/2309.11433)
- Few-Shot Learning for Medical Image Segmentation: A Review and Comparative Study, ACM Computing Surveys (2025) — [dl.acm.org/doi/10.1145/3746224](https://dl.acm.org/doi/10.1145/3746224)
- Medical Image Segmentation: A Comprehensive Review of Deep Learning-Based Methods — [pmc.ncbi.nlm.nih.gov/articles/PMC12115501](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115501/)
