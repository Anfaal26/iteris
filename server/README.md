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

See `/health`, `/models`, `/predict` for the live contract — mirrors
`iteris_ui/src/api/contract.ts`.
