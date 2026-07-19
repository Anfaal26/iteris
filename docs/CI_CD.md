# CI/CD

Iteris has three deployable pieces, each with a different pipeline. There is
no single "deploy everything" step — that's intentional, since Kaggle training
runs are manual by nature (a human reads the curves and decides to promote a
checkpoint), and the two live services deploy independently of each other.

| Piece | Where it runs | What ships it |
|---|---|---|
| Research package (`iteris/`) | Kaggle notebooks (manual) | Nothing automated — `iteris-pkg` is a Kaggle dataset you re-upload by hand after a change. CI here is sanity-only (see below), not a deploy. |
| Inference backend (`server/`) | Hugging Face Spaces (Docker) | `.github/workflows/deploy-hf-space.yml` — GitHub Action, added in this change. |
| UI (`iteris_ui/`) | Vercel | Vercel's own native GitHub integration — **not** a GitHub Action (see below). |

## CI (`.github/workflows/ci-ui.yml`, `ci-python.yml`)

Both are check-only: they run on every push to `main` and every PR, scoped by
`paths:` so an unrelated change (e.g. a notebook) doesn't trigger either.
Neither codebase has a pytest/vitest-equivalent-for-Python suite today —
`ci-python.yml` is byte-compile + import sanity (catches syntax errors, typos,
missing/renamed dependencies), not unit tests. `ci-ui.yml` runs the real
lint/vitest/build pipeline since that tooling already exists.

Nothing in CI deploys anything. A red CI check does not roll anything back —
Vercel and the HF Space deploy independently of these workflows passing or
failing (see below for why that's still safe).

## CD — Hugging Face Space (`.github/workflows/deploy-hf-space.yml`)

**This was the actual gap** flagged in the original tech-stack review: the
Space is a separate git remote from GitHub, and nothing synced `server/`
changes to it — it required a manual push or manual file upload through the HF
web UI. This workflow closes that gap: on every push to `main` touching
`server/**`, it uploads the `server/` directory's contents (`app/`,
`requirements.txt`, `Dockerfile`, `README.md`) to the Space via
`huggingface_hub.HfApi().upload_folder(...)`, which replaces the Space's file
tree and triggers an HF-side rebuild — no git-remote juggling required.

**One-time setup required** (not committed, for obvious reasons — do this in
the GitHub repo's Settings → Secrets and variables → Actions):

| Type | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | A Hugging Face access token with **write** access to the Space (create at huggingface.co → Settings → Access Tokens). |
| Variable | `HF_SPACE_ID` | `<hf-username>/<space-name>`, e.g. `Anfaal26/iteris-api` — the same Space `iteris_ui/api/infer.ts`'s `ITERIS_SPACE_URL` points at. |

Until both are set, the workflow fails loudly at the upload step with a clear
message naming which one is missing, rather than silently no-op-ing.

You can also trigger it manually from the Actions tab (`workflow_dispatch`)
without waiting for a `server/` change, e.g. to force a redeploy after
rotating a secret on the Space side.

## CD — Vercel (`iteris_ui/`)

**No GitHub Action needed or added.** Vercel deploys are driven by Vercel's
own GitHub App integration (configured once in the Vercel dashboard when the
repo is connected to a Vercel project) — it watches the repo directly and
builds/deploys on every push to `main` (and preview-deploys every PR),
entirely independent of `.github/workflows`. Adding a parallel Action-based
Vercel deploy would risk double-deploying the same commit; `ci-ui.yml` is
deliberately check-only for this reason.

If deploys ever stop happening on push, the fix is in the Vercel dashboard
(Project → Settings → Git), not in this repo's workflow files.

## What "done" looks like end-to-end

```
push to main
 ├─ touches iteris_ui/**  → ci-ui.yml runs (check only)
 │                        → Vercel's own integration builds + deploys (separately, always)
 ├─ touches server/**     → ci-python.yml's inference-server job runs (check only)
 │                        → deploy-hf-space.yml uploads server/ to the Space (deploys)
 └─ touches iteris/**     → ci-python.yml's research-package job runs (check only, no deploy)
```
