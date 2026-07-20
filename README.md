# Iteris — DRL Contour Refinement for Medical Image Segmentation

Deep-reinforcement-learning refinement of U-Net segmentation masks via **boundary-contour
deformation**, on cardiac ultrasound (CAMUS) and brain-tumour MRI (BRISC) — the only two
datasets in scope. Taylor's University capstone (PRJ63504); targeting IEEE JBHI / a MICCAI
2026 workshop.

> **Research question.** Starting from an Attention U-Net's mask, can a DRL agent recover
> additional Dice by deforming the contour — and does a **discrete** (DuelingDDQN) or
> **continuous** (TD3) agent do it better, especially as labelled data gets scarce?

---

## The idea in one picture

```
  image ──► Attention U-Net ──► coarse mask ──► contour ──► DRL agent ──► refined contour
                                  (control points + spline)   (DuelingDDQN / TD3,
                                                                angular-sector pushes)
```

RL refines the **Attention U-Net** baseline directly (single-baseline design — see
[docs/CONTEXT.md](docs/CONTEXT.md) §4/§9). A separate, deliberately lighter **LiteUNet** is
trained alongside it purely as a weaker comparison point (architecture-headroom reference),
not an RL warm-start target. The agent pushes contiguous **angular sectors** of the boundary
along their outward normals; reward is a dense per-control-point distance-to-boundary signal
(`contour_boundary`, see CONTEXT.md §6).

## Experimental design (two phases)

| Phase | Data | Asks |
|---|---|---|
| **A** | Full dataset | What does contour-refinement DRL recover over the Attention U-Net baseline when data is abundant? |
| **B** | Low-data subset (~150 imgs) | Does DRL's advantage over the supervised baseline *grow* when labels are scarce? |

A third phase (**Phase C** — swap in an archived MSA/self-attention backbone) was designed but
**formally abandoned, never implemented** — see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and
[docs/CONTEXT.md](docs/CONTEXT.md) §3 for the historical record. Only Phase A and Phase B are
real, current, and used anywhere in this repo, the evaluation notebook, or the results.

Full sizing, sources, and methodology: **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

## Repository layout

```
iteris/            Python package — all reusable logic (env, agents, training, models, metrics)
  archive/         retired paradigms kept as ablations / negative controls (+ the abandoned-Phase-C MSA backbone, never wired in)
configs/           YAML hyperparameters, nested by dataset
  CAMUS/  BRISC/   baseline (LiteUNet + AttentionResUNet) configs; DRL/ subfolder = per-class agent configs, all refining AttentionResUNet
notebooks/         thin Kaggle notebooks (import → configure → call → display)
  phaseA/, phaseB/   full-data / low-data mirror of the same notebook set: unet/ (both baselines), camus/drl/, brisc/drl/
  evaluation/      cross-run evaluation notebook — ingests every phase's output, phase-aware, never pools Phase A/B
  local/           local-GPU variants
scripts/           CLI utilities (multi-GPU runner, validation, export)
docs/              design docs — see below
server/            optional demo backend (FastAPI, self-contained — not needed to reproduce results)
iteris_ui/         optional demo frontend (React) — deployed, wired to real Phase A/B results
```

> The research core is `iteris/` + `configs/` + `notebooks/` + `scripts/`. `server/` and `iteris_ui/`
> are an optional demo product, independent of the experiments; a future cleanup may relocate both
> under an `app/` directory, but they're left at root for now to avoid churn during active runs.

## Documentation

| Doc | What it is |
|---|---|
| [docs/CONTEXT.md](docs/CONTEXT.md) | Single source of truth — what the project is, the paradigm, datasets, decisions |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Phase A/B protocol — data sizes, methodology, why (+ the abandoned Phase C record) |
| [docs/PLAN.md](docs/PLAN.md) | Strategic plan / roadmap / module responsibilities |
| [docs/SKILLS.md](docs/SKILLS.md) | Operational playbook — how it's wired, the landmines, what's open |
| [docs/TRANSCRIPT.md](docs/TRANSCRIPT.md) | Chronological engineering log |
| [docs/UI_PLAN.md](docs/UI_PLAN.md) | Demo UI brief |

## Quickstart (local)

```bash
git clone https://github.com/Anfaal26/iteris.git
cd iteris
pip install -r requirements.txt          # or: pip install -e .  (uses pyproject.toml)
```

Training runs on Kaggle (T4×2). Upload the `iteris/` package + `configs/` + `requirements.txt` as a
Kaggle Dataset named `iteris-pkg`, attach it + the data to a notebook from `notebooks/`, and run.
The agents are CPU-bound on contour rasterisation, not GPU-bound — see [docs/SKILLS.md](docs/SKILLS.md) §3.

## Status

Week ~14 of 14 (final week). Both agents (DuelingDDQN, TD3) implemented and run across CAMUS + BRISC,
**Phase A and Phase B both complete** and wired into the deployed demo UI; Phase C (MSA backbone) was
**formally abandoned**, never implemented. A phase-aware cross-run evaluation notebook
(`notebooks/evaluation/`) now ingests every run's output and is the source for the Research page's
real results. Live detail in [docs/CONTEXT.md](docs/CONTEXT.md) §8 and [docs/SKILLS.md](docs/SKILLS.md) §4.
