# Iteris UI

Frontend for **Iteris** — a medical-AI research workstation that visualises
DRL-based segmentation boundary refinement on CAMUS (cardiac ultrasound) and
BRISC (brain MRI) datasets. This package is **UI only**; all ML/training lives
in the sibling `iteris/` Python package and is reached over the API in §11.

> Research use only · Not a clinical diagnostic tool.

## Stack

React + Vite + TypeScript · Tailwind (CSS-variable tokens) · React Router ·
Three.js (landing only) · Recharts (research/library charts) · Vitest.

## Quick start

```bash
cp .env.example .env        # configure API base URL / mocks
npm install
npm run dev                 # http://localhost:5173
```

With `VITE_USE_MOCKS=true` (default) the UI runs standalone against bundled
mock data — no backend required. Point `VITE_API_BASE_URL` at the FastAPI
backend and set `VITE_USE_MOCKS=false` to use live inference.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Dev server with HMR |
| `npm run build` | Type-check + production build |
| `npm test` | Run unit tests |
| `npm run lint` | ESLint |

## Architecture (no hardcoded values)

- **Design tokens** — canonical CSS custom properties in `src/index.css`,
  mirrored for JS in `src/tokens/`, exposed to Tailwind in `tailwind.config.js`.
  Components never inline hex/px/ms.
- **API contract** — `src/api/contract.ts` is the frozen interface (spec §11).
  `src/api/client.ts` routes to live backend or `src/api/mock.ts`.
- **Content** — copy, model registry, and sample metadata live in
  `src/content/*.yaml`, not in components.
- **Config** — runtime settings resolve from env (`src/config/app.config.ts`).

## Pages (spec §3)

| Route | Page |
| --- | --- |
| `/` | Landing |
| `/research` | Research showcase |
| `/workspace` | Iteris workstation |
| `/models` | Model library |
| `/datasets` | Dataset explorer |
