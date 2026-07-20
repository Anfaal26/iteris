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
`notebooks/phaseA/unet/05_brisc_tumor_classifier.ipynb`) to enable real tumor-type
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
live in separate HF Hub model repos (`Anfaal26/iteris-drl-camus`,
`Anfaal26/iteris-drl-brisc`), pulled at runtime via
`huggingface_hub.hf_hub_download` and cached by that library — no Kaggle
credentials or downloads happen on this Space at all. Weights get into those
repos via a Kaggle notebook (`hf-link.ipynb`) that uploads training-output
datasets to HF Hub; this Space only ever reads from HF Hub.

**Registry** — `app/drl.py`'s `REGISTRY` is generated over
`{dataset} x {class} x {regime} x {algo}` (Phase A = high regime, Phase B =
low regime; both use the identical per-class contour-env hyperparameters —
regime only selects which checkpoint to load, see `_CAMUS_CLASSES` /
`_BRISC_CLASSES` in `drl.py`), 16 entries total. Each is an env-overridable
`(HF_REPO_*, HF_FILE_*)` pair with a default already pointing at the
canonical path, so no Space secrets are required unless a repo/path changes:

| Dataset | Class | Regime | Algo | HF repo | Path | Kaggle source |
|---|---|---|---|---|---|---|
| CAMUS | LV endo | High | DuelingDDQN | iteris-drl-camus | `duelingddqn/lv/high.pt` | `junkit1688/pa-camus-dueling-1-outputs` |
| CAMUS | LV epi/myo | High | DuelingDDQN | iteris-drl-camus | `duelingddqn/myo/high.pt` | `junkit1688/pa-camus-dueling-2-outputs` |
| CAMUS | LA | High | DuelingDDQN | iteris-drl-camus | `duelingddqn/la/high.pt` | `anfaalhossain/pa-camus-dueling-3-outputs` |
| CAMUS | LV endo | High | TD3 | iteris-drl-camus | `td3/lv/high.pt` | `chuachongeu/pa-camus-td3-c1-outputs` |
| CAMUS | LV epi/myo | High | TD3 | iteris-drl-camus | `td3/myo/high.pt` | `chuachongeu/pa-camus-td3-c2-outputs` |
| CAMUS | LA | High | TD3 | iteris-drl-camus | `td3/la/high.pt` | `chuachongeu/pa-camus-td3-c3-outputs` |
| BRISC | tumor | High | DuelingDDQN | iteris-drl-brisc | `duelingddqn/tumor/high.pt` | `junkit1688/pa-brisc-dueling-1-outputs` |
| BRISC | tumor | High | TD3 | iteris-drl-brisc | `td3/tumor/high.pt` | `anfaalhossain/pa-brisc-td3-outputs` |
| CAMUS | LV endo | Low | DuelingDDQN | iteris-drl-camus | `duelingddqn/lv/low.pt` | `mahfuza02/pb-camus-dueling-1-outputs` |
| CAMUS | LV epi/myo | Low | DuelingDDQN | iteris-drl-camus | `duelingddqn/myo/low.pt` | `mahfuza02/pb-camus-dueling-2-outputs` |
| CAMUS | LA | Low | DuelingDDQN | iteris-drl-camus | `duelingddqn/la/low.pt` | `mahfuza02/pb-camus-dueling-3-outputs` |
| CAMUS | LV endo | Low | TD3 | iteris-drl-camus | `td3/lv/low.pt` | `anfaalhossain/pb-camus-td3-1-outputs` |
| CAMUS | LV epi/myo | Low | TD3 | iteris-drl-camus | `td3/myo/low.pt` | `junkit1688/pb-camus-td3-2-outputs` |
| CAMUS | LA | Low | TD3 | iteris-drl-camus | `td3/la/low.pt` | `mahfuza02/pb-camus-td3-3-outputs` |
| BRISC | tumor | Low | DuelingDDQN | iteris-drl-brisc | `duelingddqn/tumor/low.pt` | `anfaalhossain/pb-brisc-dueling-outputs` |
| BRISC | tumor | Low | TD3 | iteris-drl-brisc | `td3/tumor/low.pt` | `anfaalhossain/pb-brisc-td3-outputs` |

CAMUS's 3 classes are trained as **separate agents**; a single CAMUS+algo
request fans out to all 3 registered class-agents and combines their masks
into one label map (`CLASS_PRIORITY` in `drl.py`). BRISC is single-class.

Env hyperparameters (n_points, disp_px, spline_smooth, cont_sectors, ...) are
per-checkpoint and MUST match the exact training config — see the `_CAMUS_*`
/ `_BRISC_*` dicts in `drl.py`, sourced from
`configs/CAMUS/DRL/camus_drl_c{1,2,3}.yaml` and
`configs/BRISC/DRL/brisc_drl_tumor.yaml`. Don't assume they're interchangeable
across classes/datasets — CAMUS-LA and BRISC use a different `spline_smooth`,
and BRISC's TD3 uses 12 continuous sectors vs. 16 for CAMUS.

Adding a future phase/class is a matter of: (1) upload the new checkpoint at
its own `algo/class/regime.pt` path via the same notebook pattern, (2) add it
to `_CAMUS_CLASSES` / `_BRISC_CLASSES` (a new class) or the regime tuple in
the generator loop (a new regime) — no changes to the loading logic.

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
