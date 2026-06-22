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
