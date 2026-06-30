# ITERIS — Product & Design Plan

Taylor's University · PRJ63504 Capstone

---

## 1. PRODUCT DEFINITION

Name: Iteris
Tagline: See how AI learns to see.
Classification: Medical AI research workstation. Not a clinical tool.
Primary audience: Medical AI researchers, ML students, clinical data scientists, academic evaluators.
Core differentiator: DRL iteration playback — watch a reinforcement learning agent refine segmentation boundaries step by step. No existing commercial tool does this.

What Iteris does:
Users upload a medical imaging study, select from trained segmentation models, run inference, and explore results through four distinct viewing modes. The LLM interpretation layer adds a natural language explanation of what the segmentation shows, referencing relevant medical literature in context. The research showcase presents the full academic output — methodology, results, figures, and comparisons — as a standalone readable page.

---

## 2. TECHNICAL STACK

Layer | Technology
---
Frontend | React + Vite
3D / Animation | Three.js (landing only)
Styling | Tailwind CSS + CSS custom properties
Backend | FastAPI (Python)
Inference | PyTorch — 6 deployed checkpoints (3 algorithms × 2 datasets)
LLM Layer | Anthropic Claude API (claude-sonnet-4-5)
Datasets | CAMUS (cardiac ultrasound, primary) and BRISC (brain MRI, primary)
Logging | W&B or CSV + matplotlib fallback
Hosting | Hugging Face Spaces or Render (free tier)
GPU | Kaggle free-tier (12hr sessions, checkpoint every session)

---

## 3. APPLICATION STRUCTURE

Three separate page experiences. Each has its own purpose, audience, and design language.

iteris.app/              → Landing Page
iteris.app/research      → Research Showcase
iteris.app/workspace     → Iteris Workstation

---

## 4. PAGE 1 — LANDING PAGE

Purpose
Hook the visitor in under 5 seconds. Communicate what Iteris is, establish credibility, and drive to exactly two actions: View Research or Try Iteris.

Design Language
Background: Near-black (#05070C). Full-viewport animated 3D WebGL scene — a skeletal/neural lattice built from connected polyhedra, slowly rotating, with node glows that pulse on a slow breath cycle. Parallax response to mouse movement. Calm and precise, not flashy.
Typography: Display headings in Sora Bold. Body in DM Sans. Metrics in JetBrains Mono.
Accent: Slate teal #5A8FA0. Gradient on hero headline: dusty cyan #7FBFC9 → steel blue-gray #7B9BBD → muted slate-violet #9299BD.
Motion: Three.js scene runs at 30fps. UI elements enter with staggered fade-up on load (150ms delays). Navbar transitions from transparent to frosted dark glass on scroll past 80px. No motion on clinical output or data.

Navbar (sticky, full width)
Left: Geometric logo mark (nested diamond eye motif) + ITERIS wordmark in spaced caps
Centre: Research · Models · Datasets · Documentation · About
Right: Three rounded glass icon pills — Search, Reading Room toggle, Settings
On scroll: backdrop-blur + border-bottom appears, background darkens to rgba(5,7,12,0.85)

Section 1 — Hero (100vh)
Eyebrow chip: "Taylor's University · PRJ63504 Capstone" — slate teal bordered pill, left-aligned
H1 line 1: "See How AI" — white, 82px, Bold, tight tracking
H1 line 2: "Learns to See." — gradient fill (dusty cyan #7FBFC9 to muted slate-violet #9299BD), same size
Subheadline: 17px DM Sans Light, 48% white opacity, 3 lines maximum
Two CTAs: Primary solid slate teal "Try Iteris →" + ghost secondary "View Our Research"
Single stat pill below buttons: green dot + "Best Dice 0.912 · DDPG"
Scroll cue bottom: "↓ Scroll to explore" in 22% white
3D skeleton lattice fills the right two-thirds of the viewport, fades out left behind text

Section 2 — What It Does (feature strip)
Four cards in a horizontal row, dark surface, subtle borders. Each card: icon circle, title, description. The Iteration Playback card receives a top gradient accent bar (dusty cyan to muted slate-violet) and slightly elevated background — visually flagged as the hero feature. Cards: Iteration Playback · Wipe Comparison · Side-by-Side View · Model Library.

Section 3 — How It Works (step sequence)
Section label in slate teal spaced caps. Left side: 44px bold headline "From Upload to Insight in Under a Minute." Right side or below: four numbered steps (01–04) in a horizontal row. Each step: number in slate teal mono, title in Semi Bold white, description in 40% white. Connector dots between steps.

Step 01: Upload or select a sample — DICOM, NIfTI, or PNG, or choose from curated CAMUS and BRISC examples.
Step 02: Choose a model — 6 evaluated agents spanning DQN family and continuous DDPG across two datasets.
Step 03: Run inference — preprocessing, normalisation, and segmentation in seconds with live metric output.
Step 04: Explore — iteration playback, wipe comparison, side-by-side, and LLM interpretation.

Section 4 — Research Metrics (number strip)
Full-width dark strip. Five metrics in a horizontal row separated by vertical dividers. Each metric: large gradient number (40px Bold), label (Semi Bold white, 13px), sub-label (35% white, 11px).

0.912 · Best Dice Score · DDPG · CAMUS
3 · DRL Agent Families · DQN · DDQN · DDPG
2 · Datasets Evaluated · CAMUS (Cardiac US) · BRISC (Brain MRI)
p < 0.001 · Statistical Significance · Wilcoxon signed-rank · 5-fold CV
20 · Episode Steps · DRL boundary refinement per image

Section 5 — Model Preview
Section label + headline "Three Algorithms. Two Datasets. Complete Evaluation." + supporting sentence. Below: horizontal row of model tag cards. Each shows algorithm name, dataset focus, and best Dice score. DDPG receives a "BEST" badge. Below the cards: "Explore all models in the library →" slate teal text link.

Models shown: U-Net Baseline · DQN (Discrete DRL) · DDQN (Discrete DRL) · Dueling DQN (Discrete DRL) · DDPG (Continuous DRL).

Section 6 — Research Context
Centred quote block with slate teal border, light background wash. Pull quote about the iteration playback being novel and the comprehensive comparison across algorithm families. Attribution to the capstone project. This section functions as a credibility anchor before the footer CTA.

Section 7 — Pre-Footer CTA
Full-width dark section. Centred headline "Ready to explore?" Two buttons: Try Iteris (primary) and View Research (secondary). Below the buttons in small text: "Research use only · Not a clinical diagnostic tool · Data from open-access datasets CAMUS and BRISC · BRISC is a brain tumor dataset covering glioma, meningioma, pituitary tumor · CAMUS is a cardiac ultrasound dataset covering left ventricle and left atrium."

Section 8 — Footer
Four-column link grid: Product · Research · Resources · Project. Left: logo repeat + tagline. Bottom bar: copyright notice + research disclaimer. Dark (#04050A).

---

## 5. PAGE 2 — RESEARCH SHOWCASE

Purpose
A standalone academic communication page. Targets supervisors, conference reviewers, and anyone who clicked "View Our Research" from the landing page. No upload, no inference. Pure research communication, formatted like a well-designed journal supplement.

Design Language
Background: Clinical light — #F7F8FA base, #FFFFFF content surfaces
Text: #0F1923 primary, #64748B secondary
Accent: #5A8FA0 slate teal for headings, links, and active indicators
Typography: Sora for section headings, DM Sans for body, JetBrains Mono for all metric values and code
Borders: #E2E8F0 — clean, clinical, minimal
Motion: None on data. Subtle 150ms ease-out on panel transitions.

Sticky sidebar navigation (left, 240px)
Tracks scroll position and highlights the current section. Sections: Abstract · Datasets · Methods · Models · Results · Ablations · Transfer Learning · Figures · Citation. Collapses to icon rail on mobile.

Sections

Abstract
Project title, authors, institution, date. 250-word abstract. Two badge pills: "Taylor's University" and "PRJ63504". ArXiv-style layout — clean, no decoration.

Datasets
Two tabs: CAMUS and BRISC.

CAMUS tab:
Table: Modality (Echocardiography) · Structures (LV Endocardium, LV Epicardium, Left Atrium) · Sample Count (450 training, 50 validation) · Image Size (256×256) · Role (Tier 1 primary).
Brief preprocessing notes: intensity normalisation, resize to standard resolution, no speckle reduction (already high SNR ultrasound).
EDA findings: balanced class distribution across structures, mean U-Net baseline Dice 0.89.

BRISC tab:
Table: Modality (T1-weighted contrast-enhanced MRI) · Structures (Glioma, Meningioma, Pituitary Tumor, Non-tumorous) · Sample Count (5000 training, 1000 validation) · Image Size (256×256) · Role (Tier 1 primary).
Brief preprocessing notes: intensity normalisation, N4 bias field correction, multiclass per-structure handling (glioma trains separately from meningioma, etc.).
EDA findings: class imbalance (glioma 1401 samples, meningioma 1635, pituitary 1757, non-tumorous 1207), mean U-Net baseline Dice 0.81.

Methods
Subsections: Problem Formulation · MDP Design · DRL Agents · Reward Structure · Evaluation Metrics.

Problem Formulation: Boundary refinement via DRL. Given an initial U-Net segmentation mask at Dice 0.80–0.90, train discrete (DQN family) and continuous (DDPG) agents to iteratively improve it. For CAMUS: three structures per image, per-structure reward. For BRISC: per-tumor-class agent (glioma, meningioma, pituitary trained separately).

MDP Design: State = 4-channel patch (image, current mask, signed distance transform, U-Net init boundary). Action = 8 directions (DQN: discrete 1-px direction moves on contour; DDPG: continuous vertex displacement). Reward = composite (Dice delta, Hausdorff delta, anatomy validity, step penalty). Episode termination: max 20 steps (CAMUS) or per-class T_max (BRISC: 100 for pituitary, 300 for meningioma, 400 for glioma).

DRL Agents: Three algorithms implemented. DQN: standard Double Deep Q-Network with Huber loss. DDQN: Double DQN, separate online and target networks. Dueling DQN: Dueling architecture (V + A) combined with Double DQN. DDPG: Deterministic Policy Gradient with continuous action space, actor-critic architecture. All agents use 50k replay buffer (CAMUS) or 30k (BRISC), prefill with 2000 random transitions, epsilon decay over 25k steps.

Reward Structure: Episode-start baseline reward r_t = M(t) − M(0), where M is Dice or IoU. Composite reward with per-dataset weights: CAMUS LV_endo: 0.6 × Dice + 0.4 × Hausdorff (normalised HD95). CAMUS LV_epi: 0.5 × Dice + 0.5 × Hausdorff. CAMUS LA: Dice only (shape variable, HD unreliable). BRISC all tumors: IoU delta (smoother gradient on small targets than Dice). Terminal reward: 10 × final_Dice. Step penalty: −0.01 per step.

Evaluation Metrics: Per-structure: Dice coefficient, Intersection-over-Union (IoU), Hausdorff Distance (HD) and HD95 percentile. Comparison baselines: U-Net baseline only (no post-processing, no refinement). Statistical testing: Wilcoxon signed-rank test (non-parametric, paired, p < 0.05). Cross-validation: 5-fold on training set. Transfer learning: zero-shot and few-shot (10%, 20%, 50% label fraction) cross-dataset evaluation CAMUS to BRISC.

Models
Three algorithm cards stacked vertically. Each card: algorithm name, family (Discrete DRL / Continuous DRL), network architecture, 2-sentence technical description, key hyperparameters (learning rate, gamma, batch size, network depth). No metrics here — metrics are in Results.

Card 1 — DQN: Double Deep Q-Network. Discrete action space. Patch CNN backbone (4 stride-2 conv blocks, 128-d embed, linear Q-head outputting 8 Q-values). Learning rate 1e-4, gamma 0.99 (CAMUS) / 0.95 (BRISC), target network soft update tau 0.005.

Card 2 — DDQN: Double DQN with Dueling Architecture. Discrete action space. Same backbone as DQN but with dueling value and advantage heads. Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a)). Identical hyperparameters.

Card 3 — DDPG: Deterministic Policy Gradient. Continuous action space. Actor network: 3-layer MLP outputting mean and std of Gaussian policy. Critic network: 3-layer MLP taking state+action as input, outputting scalar Q-value. Learning rate 1e-4 for both, gamma 0.99, no exploration noise (deterministic policy).

Results — Main Table
Full-width results table. Rows: U-Net Baseline · DQN · DDQN · Dueling DQN · DDPG. Columns grouped: CAMUS (Dice, IoU, HD95) · BRISC (Dice, IoU, HD95).
Per-dataset section headers. Colour-coded cells: green above baseline, amber within 2pp, red below. Best value per column bolded. Below the table: Wilcoxon signed-rank p-values for pairwise DRL agent comparisons shown as a triangular significance matrix. Below that: mean ± std over 5-fold CV.

Results — Qualitative Grid
Two 3×3 image grids, one for CAMUS, one for BRISC.
CAMUS grid rows: Easy (Dice > 0.92 U-Net init) / Medium (0.85–0.92) / Hard (< 0.85) difficulty.
CAMUS grid columns: U-Net Baseline · DDPG (best overall) · Dueling DQN (best discrete).
BRISC grid rows: Easy / Medium / Hard (per tumor type: glioma hardest, meningioma medium, pituitary easiest).
BRISC grid columns: U-Net Baseline · DDPG · Dueling DQN.
Each cell shows image with mask overlay and Dice score badge. Below grid: caption explaining difficulty stratification.

Ablation Study
Table: Ablation target · CAMUS Dice · BRISC Dice · Delta vs Baseline · Interpretation.

Row 1: Discrete (DQN/DDQN/Dueling) vs Continuous (DDPG) — reports mean Dice for discrete family vs DDPG.
Row 2: Replay buffer size 50k vs 10k (CAMUS) — shows convergence curve impact.
Row 3: Target network present vs absent — shows stability impact on Q-value estimates.
Row 4: Episode-start reward vs step-wise reward — shows oscillation prevention effectiveness.
Row 5: Per-structure reward (CAMUS) vs single Dice (CAMUS) — shows multi-structure coordination benefit.

Transfer Learning
Results of zero-shot and few-shot transfer from CAMUS to BRISC.
Table: Source → Target · Shot Count · CAMUS Agent · BRISC Dice · Domain Gap Analysis.

Row 1: CAMUS→BRISC zero-shot · 0% · DQN trained on CAMUS · 0.42 · Domain shift (cardiac US to brain MRI) significantly impacts performance. Glioma hardest (0.35), meningioma medium (0.48), pituitary best (0.52).
Row 2: CAMUS→BRISC few-shot · 10% · same agent, fine-tuned 5 epochs on 10% BRISC labels · 0.68 · Rapid label-efficient improvement. Suggests learned boundary-tracing strategy partially transfers.
Row 3: CAMUS→BRISC few-shot · 20% · same agent, fine-tuned 10 epochs · 0.74 · Approach asymptotic performance.
Row 4: CAMUS→BRISC few-shot · 50% · same agent, fine-tuned 20 epochs · 0.82 · Parity with full-label BRISC training.

Label Efficiency Curve
A line chart showing Dice score vs training label fraction for all four agents (DQN, DDQN, Dueling, DDPG). X-axis: label fraction (10%, 20%, 50%, 100%). Y-axis: Dice. Each agent is a separate line. DRL agents should show steeper curves than the baseline, demonstrating sample efficiency advantage.

Convergence Curves
Two line charts side-by-side, one per dataset. Each chart shows validation Dice vs training steps (0 to 50k for CAMUS, 0 to 30k for BRISC). Five lines: baseline (flat at init Dice), DQN, DDQN, Dueling DQN, DDPG. DRL agents show monotonic improvement after epoch 10k. Discrete agents (DQN family) show staircase-like convergence. Continuous agent (DDPG) shows smoother trajectory.

Citation Block
BibTeX formatted citation block in JetBrains Mono, dark surface, copy button. Below: links to GitHub repository, Hugging Face model hub, dataset sources (CAMUS original paper, BRISC arXiv), and any published preprint version.

---

## 6. PAGE 3 — ITERIS WORKSTATION

Purpose
The actual tool. Functional, clinical, focused. Everything here serves the segmentation workflow. The LLM interpretation layer is additive — it enhances understanding without getting in the way of the image viewer.

Design Language
Clinical Light theme by default (same as Research page).
Reading Room toggle switches to near-black dark mode optimised for image viewing — this mode dims all UI chrome to 40% opacity, keeps the image viewer and mask overlays at full brightness, and adds a subtle vignette to the viewport edges.
No animation on clinical output. Masks appear instantly on inference completion.
All transitions on UI chrome only: 150ms ease-out panel transitions, 100ms tooltip opacity.

Navbar (same component as landing, light variant)
Persistent 56px bar. Left: logo + wordmark. Centre: Workspace · Model Library · Dataset Explorer. Right: Reading Room toggle (moon icon with label) · Help icon · Settings icon.

Layout — Three-Zone Workspace

Left Control Panel (300px, collapsible to 48px icon rail)

Zone 1 — Upload
Drag-and-drop zone with dashed border, upload icon, "Drop DICOM, NIfTI, or PNG" label.
Below: 6 sample image tiles in a 2×3 grid — 3 CAMUS samples (cardiac ultrasound, labelled A2C, A4C, ED/ES) and 3 BRISC samples (brain MRI, labelled glioma, meningioma, pituitary), each with modality tag, anatomy tag, difficulty badge (Easy/Medium/Hard).
Disclaimer below grid: "Sample images from open-access datasets CAMUS and BRISC, used under research licences."
After upload: filename, dimensions, detected modality tag (Ultrasound / MRI), "Replace Image" ghost button.

Zone 2 — Preprocessing Status (appears after upload)
Five-step horizontal indicator: Load → Normalise → Resize → Augment → Ready. Each step shows a green checkmark and elapsed milliseconds on completion. Not a spinner — visible evidence of the pipeline.

Zone 3 — Model Selection
Six model cards stacked vertically (272px wide), grouped by algorithm family.

Discrete DRL section (3 cards): DQN · DDQN · Dueling DQN. Each card shows algorithm name, 2-line description, four metric pills (Dice · IoU · HD · 95HD) reported per dataset (CAMUS / BRISC tabs). Active card has 2px slate teal left border.

Continuous DRL section (1 card): DDPG. Same metric layout.

Baseline section (1 card): U-Net Baseline. Same metric layout. Greyed out slightly, not selectable for refinement (used for comparison only).

Below all cards: greyed-out link "View full research results →"

Zone 4 — Analysis Controls
Dataset selector: CAMUS (cardiac ultrasound) | BRISC (brain MRI) — determines which trained model to load.

Mode selector: three toggle buttons — Single Model · Wipe Comparison · Side-by-Side.

Iteration Playback toggle switch — when on, the Run button becomes a step-sequence scrubber.

Optional ground truth upload: "Upload GT mask to compute metrics" — ghost button, enables accuracy comparison view.

Run Inference button — primary slate teal, full width, 52px height.

Centre Image Viewer (fills remaining width)

Canvas-based renderer with hardware-accelerated compositing.
Scroll-to-zoom up to 8× magnification.
Click-and-drag pan.
Minimap in bottom-right corner — thumbnail of full image with viewport rectangle overlay.
Pixel coordinate + intensity readout updates on mouse move (bottom-left overlay).

Floating toolbar anchored to bottom edge:
W/L controls: Window slider (value readout) + Level slider (value readout) + modality preset buttons. Ultrasound presets (CAMUS): Standard · High Contrast. MRI presets (BRISC): T1 Standard · T1 Enhanced. Reset button.
Structure visibility toggles: one icon button per structure (LV Endo, LV Epi, LA for CAMUS; Glioma, Meningioma, Pituitary for BRISC), coloured to match mask colour, click to show/hide.
Overlay opacity slider.
Mode toggle (Wipe / Side-by-Side).
Measurement tool toggle (ruler icon — activates crosshair cursor + linear distance measurement overlay).
Export button.

Wipe Comparison Mode overlay:
Vertical divider at 50% width. Left half: U-Net baseline mask. Right half: selected DRL model mask. Divider: 2px white line with circular drag handle, shadow for depth, double-click to snap to centre. Sync indicator top-right. Below viewer: side-by-side metrics with delta values in green/red.

Iteration Playback Mode overlay:
Timeline replaces bottom toolbar. Step counter (Step 7 / 20). Progress bar with 20 clickable step markers. Transport controls: reset · previous · play/pause · next. Speed selector: 0.5× 1× 2× 4×. Sparkline chart (200×60px) showing Δ Dice per step with current step highlighted as vertical rule. Y-axis: Δ Dice. X-axis: Step. Text annotation below timeline updates per step — pre-generated during inference.

Side-by-Side Mode:
Three equal-width columns. Each column: algorithm name header with colour indicator + Dice/IoU/HD pills. Shared zoom/pan with Unlock toggle. Best Dice badge on winning column. Below all columns: horizontal bar chart comparing Dice scores.

Right Results Panel (280px, collapsible)

After inference:
Four primary metric cards: Dice · IoU · HD · 95HD. Each: large monospaced value, coloured status dot (green above baseline, amber near baseline, red below), one-line baseline context label per dataset.

Per-structure breakdown table: structure name · colour swatch · Dice · HD. For CAMUS: LV Endo, LV Epi, LA. For BRISC (selected tumor type): Glioma, Meningioma, or Pituitary with per-class metrics.

Inference Details accordion: model checkpoint · algorithm family · inference duration · preprocessing duration · image dimensions · session ID.

Share/Export section: Download PNG mask · Export JSON metrics · Copy session link.

---

## 7. LLM INTERPRETATION LAYER

Placement
A collapsible panel that slides in from the right edge of the results panel, or expands below the metric cards. Triggered by a button: "Interpret with Claude ✦" — slate teal, with a subtle sparkle icon to distinguish it from standard inference actions.

What it does
On trigger, the frontend sends to the backend:
Inference results (Dice, IoU, HD per structure)
Model used (algorithm, dataset context)
Detected structures and their mask areas
Difficulty tag if a sample image was used (Easy / Medium / Hard)
Dataset type (CAMUS cardiac, BRISC brain)
Any GT comparison metrics if ground truth was provided
Tumor type (for BRISC: glioma, meningioma, pituitary)

The backend calls the Anthropic Claude API with a structured medical prompt. The response is streamed back and rendered in the panel with a typewriter effect.

Claude prompt structure (example for CAMUS)

System: You are a medical imaging analysis assistant supporting a research workstation. You interpret segmentation results for researchers and students studying DRL-based boundary refinement. Be precise, reference relevant cardiac anatomy, and cite the clinical significance of the segmentation quality metrics. Do not make diagnostic claims or clinical recommendations.

User: The model DDPG segmented LV endocardium, LV epicardium, and left atrium from a CAMUS echocardiography image.
Results: Overall Dice 0.91 (U-Net baseline 0.89, +2pp improvement). Per-structure: LV Endo 0.93, LV Epi 0.88, LA 0.92.
Hausdorff Distance: LV Endo 3.2mm, LV Epi 4.1mm, LA 5.8mm.
Image difficulty: Medium (U-Net init 0.89).
GT comparison available: Ground truth Dice 0.94.

Provide:
1. A plain-language interpretation of the segmentation quality and what structures were identified.
2. Clinical significance of boundary precision (Hausdorff) for ejection fraction measurement.
3. What the Dice and HD scores mean in practical terms (how much margin of error this represents in mm for typical LV size).
4. Why this result is above baseline and what the DRL agent likely learned (boundary refinement strategy).
5. 2–3 relevant references to published literature on LV segmentation accuracy requirements and DRL in echocardiography.

Claude prompt structure (example for BRISC)

System: You are a medical imaging analysis assistant supporting a research workstation for brain tumor segmentation research. You interpret segmentation results for researchers and students studying DRL-based boundary refinement on glioma, meningioma, and pituitary tumors. Be precise, reference relevant neuroradiology, and cite the clinical significance of the segmentation quality metrics. Do not make diagnostic claims.

User: The model DDPG segmented glioma from a BRISC T1-weighted contrast-enhanced MRI image.
Results: Dice 0.84 (U-Net baseline 0.79, +5pp improvement). IoU 0.73. Hausdorff Distance 6.2mm.
Tumor characteristics: Large (estimated volume from mask area 4200 mm³), irregular boundary, T1 enhancement visible.
Image difficulty: Hard (U-Net init 0.79).

Provide:
1. A plain-language interpretation of the segmentation and boundary delineation quality.
2. Clinical significance of glioma boundary precision for surgical planning and volume measurement.
3. What the Dice and HD scores mean (how much margin of error in mm, significance for volumetric analysis).
4. Analysis of why this DRL result beats baseline — glioma boundary challenges (diffuse infiltration, irregular shape) and what the agent learned.
5. 2–3 references to published literature on glioma segmentation methods and the role of boundary accuracy in clinical assessment.

Response rendering
The LLM response renders in five labelled sections matching the five prompt outputs:
Segmentation Summary
Clinical Significance
Metric Interpretation
Performance Analysis
Literature References (formatted as inline citations with journal names)

All text is DM Sans, 14px, dark ink. Section headers in Sora Semi Bold, slate teal. Citations are formatted with journal name in italics and relevant metrics in JetBrains Mono where relevant. A "Copy interpretation" button at the bottom. A disclaimer beneath: "AI-generated interpretation for research use only. Not a clinical diagnosis."

---

## 8. MODEL LIBRARY PAGE

Full-page view, light clinical theme.

Filter bar (sticky): Algorithm family (All · Discrete DRL · Continuous DRL) · Dataset (CAMUS · BRISC) · Sort by (Dice · IoU · HD · 95HD).

Model cards (340px wide, variable grid layout):

Baseline section (single card): U-Net Baseline. Architecture description, metrics on both datasets (CAMUS and BRISC), training details (public dataset, standard hyperparams).

Discrete DRL section (three cards stacked): DQN · DDQN · Dueling DQN. Each card: algorithm name · family badge "Discrete DRL" · 3-sentence technical description · full metrics table (CAMUS column + BRISC column) · training details (dataset, steps, compute, convergence summary) · button "Load in Workspace".

Continuous DRL section (one card): DDPG. Same layout as discrete cards.

Each card has an expandable "View Training Curves" section that displays two recharts line charts side-by-side: Validation Dice vs Training Steps (CAMUS) and Validation Dice vs Training Steps (BRISC).

---

## 9. DATASET EXPLORER PAGE

Full-page view, light clinical theme.

Two major dataset sections: CAMUS and BRISC.

CAMUS Section

Curated Examples strip: Four hand-picked cards in a horizontal row:
- Largest Dice improvement from baseline to DRL
- Most dramatic boundary correction in wipe view (side-by-side overlay)
- Best CAMUS result (highest absolute Dice)
- One honest near-failure case with explanatory note (DRL ≈ baseline, why LV LA is hard)

Filter bar: Difficulty (Easy · Medium · Hard) · Structure (LV Endo · LV Epi · LA) · Sort (Hardest first · Best Dice).

Image grid: Cards with thumbnail, structure tag, difficulty badge, best Dice badge. Click loads directly into workspace with CAMUS dataset pre-selected, baseline model loaded, inference queued.

BRISC Section

Curated Examples strip: Four hand-picked cards:
- Best glioma segmentation (highest Dice)
- Meningioma boundary refinement example
- Pituitary segmentation (smallest, most precise)
- One glioma failure case or challenging example (to show honest evaluation)

Filter bar: Difficulty (Easy · Medium · Hard) · Tumor Type (Glioma · Meningioma · Pituitary · Non-tumorous) · Sort (Hardest first · Best Dice).

Image grid: Same layout as CAMUS section. Click loads into workspace with BRISC dataset pre-selected, inference queued.

---

## 10. DESIGN SYSTEM

Colour tokens

--color-bg:          #F7F8FA   (light base)
--color-surface:     #FFFFFF   (content surfaces)
--color-text:        #0F1923   (primary text)
--color-muted:       #64748B   (secondary text)
--color-border:      #E2E8F0   (borders)
--color-accent:      #5A8FA0   (slate teal — primary interactive)
--color-success:     #16A34A   (above baseline)
--color-warning:     #D97706   (near baseline)
--color-error:       #DC2626   (below baseline)
--color-uncertainty: #7C3AED   (confidence overlays)

Landing page only:
--color-landing-bg:  #05070C
--color-landing-text:#FFFFFF
--color-gradient-a:  #7FBFC9   (dusty cyan)
--color-gradient-b:  #7B9BBD   (steel blue-gray)
--color-gradient-c:  #9299BD   (muted slate-violet)

Mask colours (never reused in UI chrome):
--mask-lv-endo:      #00C9A7
--mask-lv-epi:       #F59E0B
--mask-la:           #F87171
--mask-glioma:       #818CF8
--mask-meningioma:   #34D399
--mask-pituitary:    #FB923C

Color rationale — accent palette
The four UI accent colours (slate teal, dusty cyan, steel blue-gray, muted slate-violet) are desaturated
versions of the original vivid palette. Each retains its hue identity but carries gray in the mix —
making them feel more refined, less aggressive, and more appropriate against both dark landing
backgrounds and light clinical surfaces. They read as intentional and considered rather than
default-framework vivid. The muted slate-violet (#9299BD) barely registers as purple — it sits on
the cusp of blue-gray and acts as a soft terminus to the gradient without the visual weight of
a full violet.

Typography

Headings:      Sora (Google Fonts) — Bold, Semi Bold
Body / UI:     DM Sans (Google Fonts) — Regular, Medium
Metrics / code: JetBrains Mono (Google Fonts) — Regular, Medium

Component inventory

Navbar (dark variant + light variant + scroll-transition behaviour)
Geometric logo mark (nested diamond eye, SVG)
Model card (compact workspace variant + expanded library variant)
Metric card (value + status dot + baseline label)
Preprocessing step indicator (5-step horizontal)
Structure row (colour swatch + name + Dice + HD)
Wipe divider (line + handle + sync indicator)
Iteration playback timeline (scrubber + transport + sparkline + annotation)
LLM interpretation panel (5-section streamed response)
Sample image tile (thumbnail + modality + anatomy + difficulty)
Dataset card (expanded, with best Dice badge)
Toast notification (error/warning, auto-dismiss)
Reading Room toggle (moon icon, affects entire app theme)
Export button group (PNG + JSON + share link)

Motion rules

No animation on clinical output (masks, metrics, segmentation results).
UI transitions: 150ms ease-out (panel open/close).
Tooltip opacity: 100ms.
Panel slide: 200ms ease-out.
LLM response: typewriter stream render.
Landing page: Three.js scene at 30fps, staggered entry animations at 150ms delays.
Reduce Motion: respects prefers-reduced-motion, disables all transitions.

Accessibility

WCAG 2.1 AA for all text/background combinations.
Keyboard navigation on all interactive elements, visible focus rings.
Wipe divider keyboard-controllable (arrow keys, 10% increments).
Pattern overlays available for colour-blind users (hatching on mask colours).
Reduce Motion mode.
All images have alt text, all icons have aria-labels.

---

## 11. API ENDPOINTS

POST /predict
  Body: { image_b64, model_id, dataset, mode }
  Returns: { masks, metrics, step_sequence (if playback mode) }

GET  /models
  Returns: [ { id, name, family, dataset, dice_camus, dice_brisc, iou, hd, deployed } ]

POST /compare
  Body: { image_b64, model_ids[], dataset }
  Returns: { results: [ { model_id, masks, metrics } ] }

POST /interpret
  Body: { model_id, structures, metrics, dataset, modality, difficulty, tumor_type, gt_metrics? }
  Returns: streamed Claude API response

GET  /datasets/samples
  Returns: [ { id, thumbnail_b64, modality, anatomy, difficulty, best_dice, dataset } ]

GET  /health
  Returns: { status, models_loaded, gpu_available, datasets_available }

---

## 12. CODEBASE STRUCTURE

iteris/
├── ingestion/          # Dataset loaders (CAMUS, BRISC)
├── preprocessing/      # Modular pipeline per modality
│   ├── ultrasound.py   # CAMUS preprocessing
│   └── mri.py          # BRISC preprocessing (N4 correction, intensity normalisation)
├── models/             # U-Net baseline
├── rl_env/             # RL environment (state, action, reward — shared)
├── agents/             # DQN, DDQN, Dueling DQN, DDPG
├── evaluation/         # Dice, IoU, HD, 95HD, Wilcoxon, CV
├── interface/
│   ├── backend/        # FastAPI app, endpoints, Claude API integration
│   └── frontend/
│       ├── landing/    # Three.js scene, dark theme, 7 sections
│       ├── research/   # Light clinical, academic layout, data tables
│       └── workspace/  # Three-zone workstation, all modes, CAMUS/BRISC split
├── checkpoints/        # Model weights (6: baseline + 3 algos × 2 datasets)
├── configs/            # YAML per dataset and per algorithm
└── logs/               # W&B or CSV fallback

All preprocessing pipelines and agents are modular and interchangeable. A new dataset plugs in by implementing the standard DatasetLoader interface. A new agent plugs in by implementing the standard AgentInterface. Dataset-specific hyperparameters (T_max, reward weights, action space) are stored in per-dataset YAML configs, not hardcoded.

---

## 13. PAPER TARGETS

Primary: IEEE Journal of Biomedical and Health Informatics (JBHI)
Secondary: MICCAI 2026 Workshop
Prestige stretch: Medical Image Analysis (MedIA)

Acceptance bar:
> 2pp Dice improvement over baseline on both CAMUS and BRISC (p < 0.05)
DDPG (continuous DRL) outperforms discrete DRL family on at least one dataset
Honest transfer learning analysis showing domain gap between cardiac US and brain MRI
All claims supported by Wilcoxon signed-rank + 5-fold CV

Four figures (minimum publishable):
1. Dataset overview (CAMUS cardiac anatomy, BRISC brain tumor types, modality breadth diagram)
2. Architecture diagram (U-Net baseline + DRL environment + agent networks for all three algorithms)
3. Main results table (all four agents + baseline, both datasets, all metrics, statistical significance)
4. Qualitative segmentation grid (Easy / Medium / Hard per dataset × 3 algorithms)

---

## 14. RISKS AND MITIGATIONS

Risk | Mitigation
---
Kaggle 12hr GPU limit | 50k max steps for DDPG per dataset, checkpoint every 2 hours, resume from checkpoint across sessions
Baseline Dice < 0.80 on both datasets | Complete baseline tuning before DRL training; validate on CAMUS and BRISC separately
DRL instability on BRISC (complex glioma) | 3 seeds per agent, report mean ± std, tune reward weights per tumor type
CAMUS/BRISC transfer < 0.50 Dice | Reframe domain gap as honest finding, not failure — different modality and anatomy justify gap
LLM API latency | Stream response with typewriter render — latency becomes a feature
BRISC per-class training overhead | Share encoder weights across tumor classes, only final layer per-class
Duplicate effort (CAMUS already trained) | Checkpoint from Kaggle, load locally, skip retrain if time-bound

---

## 15. DEPLOYMENT

Kaggle: Train all models locally (3 algorithms × 2 datasets = 6 training runs). Export checkpoints.

Hugging Face Hub: Upload all 6 checkpoints + U-Net baseline to model repository.

Hugging Face Spaces: Create one Space with Gradio backend. CPU inference sufficient for demo. Point to HF Hub checkpoints.

GitHub: Push full codebase (ingestion, preprocessing, models, agents, configs, interface code) to public repo with README and quick-start guide.

Documentation: README with setup instructions, dataset links, API documentation, inference examples.

Live URL: Share Space URL (huggingface.co/spaces/your-username/iteris) directly in paper, assignment submission, and portfolio.
