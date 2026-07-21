# Iteris — Detailed Evaluation Report

> Source material for the capstone paper's Results/Discussion sections. Grounded entirely in the
> real 2026-07-20 Kaggle evaluation run's exported data (16 DRL run JSONs from
> `outputs/DRL Outputs-20260720T133023Z-1-001.zip` + `evaluation_outputs/master_comparison.csv`
> from `outputs/results (2).zip`) — no numbers in this document are estimated, interpolated, or
> read off a chart. **Written 2026-07-21.**

---

## 0. Critical data-integrity finding: the exported phase labels are inverted

Before any result can be trusted, this has to be flagged and corrected, because it silently
changes which regime every DRL number belongs to.

**Finding:** the 16 DRL run folders in the real Kaggle export are named `..._phase_a/` and
`..._phase_b/`. Cross-referencing each run's own `init_dice_mean` (the U-Net baseline's Dice on
that same test set, as measured by the DRL warm-start) against the actual U-Net baseline Dice
values in `master_comparison.csv` shows, **for all 4 target classes without exception**, that the
folder labeled `phase_b` matches the strong `AttentionResUNet` baseline almost exactly, while
`phase_a` is always markedly lower — consistent with `phase_a` being the ~150-sample low-data
run and `phase_b` being the full-data run:

| Class | folder `phase_a` init Dice | folder `phase_b` init Dice | AttentionResUNet (master CSV) | LiteUNet (master CSV) |
|---|---|---|---|---|
| CAMUS/LV_endo | 0.9129 | 0.9356 | 0.9360 | 0.9105 |
| CAMUS/LV_epi | 0.8166 | 0.8621 | 0.8690 | 0.8090 |
| CAMUS/LA | 0.8526 | 0.8962 | 0.8935 | 0.8214 |
| BRISC/tumor | 0.7555 | 0.8766 | 0.8695 | 0.8191 |

`folder phase_b` is within 0.001–0.007 of `AttentionResUNet` in every single class — far closer
than random chance — while `folder phase_a` sits well below it every time, in BRISC's case even
below the (independently weaker) `LiteUNet` number. This cannot be coincidence across 4/4 classes.

**Conclusion, and the correction applied throughout the rest of this report:**

> **Folder `phase_a` = TRUE Phase B (low-data, ~150-sample regime).**
> **Folder `phase_b` = TRUE Phase A (full-data regime).**

The labels are backwards relative to the project's actual experimental design (CONTEXT.md §3).
This is very likely an error introduced when the output folders were named/exported/zipped on the
Kaggle side (not a bug in the evaluation notebook — the notebook's phase detection, fixed this
session in commit `9ceea97`, faithfully parses whatever the folder says; the folder itself is
what's wrong here). **Action item for the source data:** rename or re-export the Kaggle DRL output
folders with the correct phase suffix before any further run, so this correction doesn't need to
be re-derived by hand each time. Every table below uses the **corrected** (true) phase.

**A second, related limitation:** the U-Net baseline rows in `master_comparison.csv` are tagged
`phase=Unknown` — their own `label_frac` didn't make it into that particular summary export. The
values line up with what's expected (AttentionResUNet ≈ the corrected true-Phase-A DRL init Dice,
LiteUNet noticeably lower in 3/4 classes) but this is a plausibility check, not a confirmed label —
treat any "Phase A/B AttentionResUNet baseline Dice" figure below as **most likely correct, not
certain**, since the export itself doesn't say so directly.

---

## 1. What was evaluated

- **Datasets/classes:** CAMUS (LV_endo, LV_epi, LA) and BRISC (tumor, generic pooled class).
- **Agents:** DuelingDDQN (discrete, 18 angular-sector actions) and TD3 (continuous, per-sector
  displacement) — both operating on `ContourRefineEnv`, refining the **AttentionResUNet** baseline
  directly (single-baseline design; see CONTEXT.md §4/§9). `LiteUNet` is a separate, weaker
  comparison baseline in this run's data — it is **not** what either agent refines, and is **not**
  used as a DRL comparison point anywhere below, precisely to avoid the false comparison flagged
  going into this report (comparing a low-data LiteUNet number against a full-data DRL number
  would overstate or understate DRL's effect for reasons that have nothing to do with DRL).
- **Regimes:** Phase A (full data) and Phase B (~150-sample low-data), corrected per §0.
- **Metric used for the headline verdict:** `value_floored_dice` — the deployable, GT-free "do no
  harm" selection (Pillar 1: never deploys a state valued below the initial state, using the
  agent's own value estimate, not ground truth). This is the number that would actually ship;
  `final_dice` (the raw, un-floored end-of-episode Dice) is reported alongside it since it shows
  what the floor is protecting against.
- **What this run does NOT include:** the BRISC tumor-type classifier (no classifier summary was
  attached to this export) and per-patient significance testing (no `*_test_scores.csv` was
  attached either — see §6). Both are separate, already-built parts of the evaluation notebook;
  they simply weren't in this particular data drop.

---

## 2. Headline quantitative verdict: no run beats its baseline

Using the same ±0.005 Dice tie band the evaluation notebook itself uses (a run within that band
of its baseline is a **tie**, not noise-chasing):

| Agent | Wins | Ties | Losses | n |
|---|---|---|---|---|
| DuelingDDQN | **0** | 7 | 1 | 8 |
| TD3 | **0** | 4 | 4 | 8 |

**Zero wins, for either agent, in either phase, across all 4 classes.** Every one of the 16 runs'
value-floored deployable Dice is at or below its own AttentionResUNet baseline. The value-floor
(Pillar 1) is doing real work here — it's why most runs land as a tie rather than a loss — but a
safety net that prevents harm is not the same claim as "DRL improves the mask," and this report
does not make that claim. **Numbers-wise, the honest answer to "does contour-refinement DRL
improve on the deployed baseline" is no, not in this data.**

### 2.1 Full per-run table (corrected phases)

| Dataset/Class | Agent | Phase | Init Dice | Final Dice (raw) | Deployed Dice (value-floored) | Δ deployed | HD95 init→final | Δ IoU | Δ BIoU |
|---|---|---|---|---|---|---|---|---|---|
| CAMUS/LV_endo | DuelingDDQN | A (full) | 0.9356 | 0.9301 | 0.9338 | −0.0018 | 5.71→6.05 | −0.0095 | −0.0215 |
| CAMUS/LV_endo | TD3 | A (full) | 0.9356 | 0.9205 | 0.9241 | −0.0115 | 5.71→6.55 | −0.0265 | −0.0630 |
| CAMUS/LV_endo | DuelingDDQN | B (low) | 0.9129 | 0.9106 | 0.9116 | −0.0012 | 8.01→8.19 | −0.0037 | −0.0049 |
| CAMUS/LV_endo | TD3 | B (low) | 0.9129 | 0.9068 | 0.9128 | −0.0001 (tie) | 8.01→8.27 | −0.0106 | −0.0204 |
| CAMUS/LV_epi | DuelingDDQN | A (full) | 0.8621 | 0.8592 | 0.8612 | −0.0009 | 6.59→6.74 | −0.0045 | −0.0035 |
| CAMUS/LV_epi | TD3 | A (full) | 0.8621 | 0.8354 | 0.8420 | −0.0201 | 6.59→8.58 | −0.0408 | −0.0309 |
| CAMUS/LV_epi | DuelingDDQN | B (low) | 0.8166 | 0.8157 | 0.8163 | −0.0003 | 10.02→10.06 | −0.0014 | −0.0006 |
| CAMUS/LV_epi | TD3 | B (low) | 0.8166 | 0.7891 | 0.8022 | −0.0144 | 10.02→12.36 | −0.0394 | −0.0140 |
| CAMUS/LA | DuelingDDQN | A (full) | 0.8962 | 0.8859 | 0.8937 | −0.0025 | 7.48→7.96 | −0.0162 | −0.0302 |
| CAMUS/LA | TD3 | A (full) | 0.8962 | 0.8777 | 0.8883 | −0.0080 | 7.48→8.31 | −0.0306 | −0.0536 |
| CAMUS/LA | DuelingDDQN | B (low) | 0.8526 | 0.8398 | 0.8518 | −0.0008 | 11.26→12.40 | −0.0183 | −0.0160 |
| CAMUS/LA | TD3 | B (low) | 0.8526 | 0.8315 | 0.8515 | −0.0010 | 11.26→11.71 | −0.0324 | −0.0366 |
| BRISC/tumor | DuelingDDQN | A (full) | 0.8766 | 0.8177 | 0.8672 | −0.0094 | 6.60→9.09 | −0.0922 | −0.1194 |
| BRISC/tumor | TD3 | A (full) | 0.8766 | 0.8523 | 0.8743 | −0.0023 | 6.60→6.95 | −0.0405 | −0.1398 |
| BRISC/tumor | DuelingDDQN | B (low) | 0.7555 | 0.7475 | 0.7526 | −0.0029 | 14.50→14.73 | −0.0096 | −0.0076 |
| BRISC/tumor | TD3 | B (low) | 0.7555 | 0.7368 | 0.7540 | −0.0015 | 14.50→14.56 | −0.0299 | −0.1022 |

---

## 3. Discrete vs continuous (DuelingDDQN vs TD3)

**DuelingDDQN degrades the baseline less than TD3, on average, across every metric measured:**

| Agent | Mean Δ deployed Dice | Mean Δ IoU | Mean Δ BIoU |
|---|---|---|---|
| DuelingDDQN | **−0.0025** | **−0.0194** | **−0.0255** |
| TD3 | −0.0074 | −0.0313 | −0.0576 |

Head-to-head, matched by (class, phase): **DuelingDDQN is the less-harmful agent in 6 of 8 matched
pairs.** TD3 is less-harmful only for BRISC/tumor (both phases) and CAMUS/LV_endo under low data —
everywhere else DuelingDDQN's deployed Dice sits closer to (or exactly at) its baseline.

**Interpretation, not spin:** this is consistent with the project's own structural expectation
(CONTEXT.md §9) — DuelingDDQN has a learned, explicitly-taught STOP action (the optimal-stopping
bonus, §6 of CONTEXT.md) plus the discrete action space's inherently smaller excursion per step;
TD3 has no equivalent learned stop and relies on the newer (2026-07-16, synthetic-verified-only at
the time of writing) action-magnitude auto-stop and the `fail_thresh` safety net. The larger TD3
degradation, especially the much larger **BIoU** drop (−0.058 mean vs −0.026 for DuelingDDQN, and
as much as −0.14 on BRISC), is consistent with a continuous actor over-perturbing an
already-reasonable contour when it has no strong mechanism to recognize "stop, this is good enough."

**BRISC is the one dataset where TD3 is actually the safer choice** in this data (both phases) —
worth investigating separately rather than treating the DuelingDDQN-favouring pattern as universal;
BRISC's tumor targets are small (~1.7% of image area per SKILLS.md), and continuous per-sector
displacement may cope better with that scale than 8 discrete angular sectors.

---

## 4. High-data (Phase A) vs low-data (Phase B)

| Phase | Mean Δ deployed Dice | Mean Δ IoU | Mean Δ BIoU | Mean HD95 (init → final) |
|---|---|---|---|---|
| Phase A (full-data) | −0.0071 | −0.0326 | −0.0577 | 6.60 → 7.53 |
| Phase B (low-data) | **−0.0028** | **−0.0182** | **−0.0253** | 10.95 → 11.54 |

**The average degradation is smaller under low data than under full data** — Phase B's mean Δ
deployed Dice (−0.0028) is about a third the size of Phase A's (−0.0071), and the same pattern
holds for IoU and BIoU.

**This is NOT evidence that DRL's relative advantage grows under data scarcity** — the hypothesis
the project's own literature review (EXPERIMENTS.md §1.1) was testing for. Both regimes are
negative; neither shows DRL actually improving on the baseline. What this data shows is narrower
and should be reported as such: **DRL does *less harm*, on average, when the baseline it's
refining is already weaker** (Phase B's AttentionResUNet baseline Dice is itself notably lower
than Phase A's, per §0's table — e.g. BRISC 0.7555 vs 0.8766). A plausible mechanism: the
value-floor has more headroom to matter when the starting point is already imperfect, and/or a
weaker baseline's masks may have larger, "easier" errors that a short episode can partially avoid
making worse. This is a hypothesis for the paper to state carefully, not a confirmed causal
finding — the sample size here is 4 classes × 2 agents per phase, not enough for a real
significance test on this specific comparison (see §6).

Win/tie/loss by phase (same ±0.005 band): DuelingDDQN is 3-tie/1-loss under full data and a clean
4-tie/0-loss under low data. TD3 is 1-tie/3-loss under full data and 3-tie/1-loss under low data.
Both agents' loss count drops under low data, reinforcing the pattern above.

---

## 5. Does DRL actually refine the mask? Numbers say no; qualitatively, by eye, sometimes yes on shape

**Quantitatively: no.** Section 2–4 already state this plainly and it bears repeating as the
single clearest sentence for the paper: **none of the 16 real runs improved the deployed Dice
beyond noise, and boundary-precision metrics (BIoU, HD95) got measurably *worse* on average in
every phase/agent combination** — including cases where Dice itself looked close to flat (e.g.
BRISC/tumor DuelingDDQN Phase A: Dice moved only −0.0094 but BIoU dropped −0.1194 and HD95 grew
from 6.60px to 9.09px). **A near-tied Dice number can hide a real loss of boundary quality** — the
paper should not report Dice alone as "no change" without also reporting BIoU/HD95 for the same run.

**Qualitatively: the exported comparison-grid images (Section 9/9a of the evaluation notebook,
`{model}_comparison.png` per run) show, by eye, refined masks that are visually recognizable
approximations of the true anatomical/tumor shape in most of the BEST- and MEDIAN-gain rows** —
smoother, more contour-like boundaries than a raw pixel mask might show, and generally the right
size/location/shape for the target structure (crescent/annular shapes for CAMUS LV classes,
elongated shapes for LA, round/oval masses for BRISC tumor). The WORST-gain rows show visibly
larger deviations — boundary drift, partial coverage, or shape distortion — consistent with the
quantitative degradation.

**This qualitative observation must be reported with real hedging, not as a second quantitative
result:** it comes from a visual read of a handful of exported comparison grids in this session,
not a blinded, structured, or reader-scored assessment. It is a legitimate and honest thing to
put in a paper as "informal visual inspection suggests the refined contours remain
anatomically/morphologically plausible even where they don't improve on standard overlap metrics"
— but it should not be dressed up as a rigorous qualitative evaluation. **If the paper wants a real
qualitative result** (e.g. a table of "N/M samples judged visually superior by an independent
rater"), that needs an actual structured rating pass — a natural next step given the evaluation
notebook's Section 9a already groups every model's/agent's images by (dataset, class, phase) for
exactly this kind of review, and the `iteris/refinement_viz.py` patch landed this session (commit
`9ceea97`) now labels each future comparison-plot row with its patient ID, so a **future** Kaggle
re-run would support matching the *same* patient across models/agents for a genuinely paired
qualitative comparison, which the current export does not (each agent's best/median/worst rows
are its own independently-picked patients — see the evaluation notebook's Section 9a honesty note).

---

## 6. What this report deliberately does NOT claim

Per the explicit instruction to avoid overstated comparisons:

- **No LiteUNet-vs-DRL comparison anywhere above.** LiteUNet is not the DRL target (single-baseline
  design) and comparing e.g. a Phase B LiteUNet number against a Phase A DRL number would combine
  two unrelated variables (weaker architecture *and* less data) into one misleading "DRL looks
  amazing" or "DRL looks terrible" number depending on direction — neither would be a real finding.
  LiteUNet's only legitimate role in this project is the separate architecture-headroom comparison
  in the evaluation notebook's Section 3, which this report does not repeat.
- **No claim about the BRISC tumor-type classifier** — it wasn't part of this data export.
- **No formal significance test (Wilcoxon/Bonferroni)** — that requires per-patient paired data
  (`*_test_scores.csv` for U-Net, or a per-sample DRL export), neither of which was attached to
  this run. The 16-run table above is a census of every available run, not a sample — but with
  n=4 classes per phase/agent cell, it is too small to support a hypothesis test with real power,
  and the paper should not present p-values computed on it.
- **No claim that Phase B "helps DRL more"** — see §4's explicit caveat: less harm is not more help.

---

## 7. Recommendations for the actual capstone write-up

1. **Lead with the honest quantitative finding** (§2): the value-floor makes DRL safe (mostly
   ties, never a clear loss beyond BRISC/LV_epi-TD3 outliers) but not yet beneficial. Frame the
   contribution as "a safe, deployable contour-refinement mechanism that does not yet close a
   measurable Dice gap" rather than overselling a positive result that isn't in this data.
2. **Report BIoU/HD95 alongside Dice for every headline number** — §5's finding that boundary
   quality can degrade even when Dice looks flat is one of the more interesting, defensible things
   in this dataset and is easy to miss if only Dice is reported.
3. **Discrete vs continuous**: DuelingDDQN is the safer default; note BRISC as the counter-example
   and discuss why (small-target volatility, §3) rather than presenting one agent as universally
   better.
4. **Before writing the Phase A/B section**, get the Kaggle export folder-naming fixed (§0) so
   future runs don't need this correction re-derived by hand, and ideally re-attach a run with
   `label_frac` correctly present on the U-Net baseline JSONs so the "Unknown"-phase caveat in §0
   goes away too.
5. **If a qualitative section is wanted for the paper**, budget for an actual structured rating
   pass (even 2–3 independent raters scoring best/median/worst samples) rather than relying on the
   informal read in §5 — the infrastructure for it (Section 9a's grouped gallery, the
   patient-ID-labeled future exports) is already in place.
