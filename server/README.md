---
title: Iteris Inference API
emoji: 🫀
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

FastAPI inference backend for [Iteris](https://github.com/Anfaal26/iteris). Serves
the trained Attention Res-U-Net checkpoints (CAMUS cardiac ultrasound, BRISC
brain MRI) pulled from `HF_REPO_CAMUS` / `HF_REPO_BRISC` at startup.

Set `CORS_ORIGINS` (comma-separated) as a Space secret/variable to restrict
access to your deployed frontend's origin, e.g.
`https://iteris.vercel.app,http://localhost:5173`.

Optional: set `HF_REPO_BRISC_CLASSIFIER` to a repo containing
`brisc_tumor_classifier_best.pt` (see `iteris/classifier.py` and
`notebooks/05_brisc_tumor_classifier.ipynb`) to enable real tumor-type
labels (glioma/meningioma/pituitary) on BRISC predictions instead of the
"Tumor (unclassified)" placeholder. Unset = falls back gracefully.

Optional: set `ANTHROPIC_API_KEY` (as a Space **secret**, not a public variable)
to enable `/interpret` — streams a Claude-generated explanation of each
prediction's metrics/structures in the format `LLMInterpretationPanel.tsx`
expects. Unset = `/interpret` returns 501 and the "Interpret with Claude"
button's request fails (UI shows nothing further). Optionally override the
model via `ANTHROPIC_MODEL` (defaults to `claude-sonnet-4-6`).

See `/health`, `/models`, `/predict` for the live contract — mirrors
`iteris_ui/src/api/contract.ts`.

## DRL contour-refinement (`/infer`)

`/predict` only ever serves the deployed U-Net baseline. DRL model ids
(DuelingDDQN, TD3, …) are served by `POST /infer`, keyed by
`(dataset, modelFamily, algo, regime)` — see `app/drl.py`'s `REGISTRY` dict.

Checkpoints follow the **same pattern as the Attention U-Net baselines**: they
live in a separate HF Hub model repo, `Anfaal26/iteris-drl-camus`, pulled at
runtime via `huggingface_hub.hf_hub_download` and cached by that library — no
Kaggle credentials or downloads happen on this Space at all. Weights get into
that repo via a Kaggle notebook (`hf-link.ipynb`) that uploads training-output
datasets to HF Hub; this Space only ever reads from HF Hub.

**Current registry entry**: CAMUS class 1 (LV endocardium), high regime,
DuelingDDQN — sourced from the Kaggle dataset
[`junkit1688/pa-camus-dueling-1-outputs`](https://www.kaggle.com/datasets/junkit1688/pa-camus-dueling-1-outputs),
uploaded to `Anfaal26/iteris-drl-camus` at `duelingddqn/lv/high.pt`.

Both `HF_REPO_CAMUS_DRL` and `HF_FILE_CAMUS_LV_HIGH_DUELINGDDQN` have defaults
baked in matching that path, so **no Space secrets are required** for this
entry — only overridable if the repo/path ever changes.

Adding class 2/3, the low regime, or TD3 later is a matter of: (1) upload the
new checkpoint to `Anfaal26/iteris-drl-camus` at its own `algo/class/regime.pt`
path via the same notebook pattern, (2) add one `DrlEntry` to `REGISTRY` in
`app/drl.py` — no changes to the loading or request-handling logic.

**Frontend wiring**: the Vercel app never talks to this Space's `/infer`
directly — it calls its own same-origin `/api/infer` serverless function
(`iteris_ui/api/infer.ts`), which holds this Space's URL (`ITERIS_SPACE_URL`)
and an optional bearer token (`ITERIS_SPACE_TOKEN`) as **Vercel** environment
variables, and proxies the request server-side.

**Cold starts**: free-tier Spaces sleep after inactivity and take roughly
30 seconds to a minute to wake on the first request after idling. `/infer`
does not add its own timeout beyond the platform default, so a cold start
shows up as a slow response, not a failure — the frontend should render a
loading state for DRL runs rather than treating a slow `/api/infer` call as
an error.
