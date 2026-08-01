"""
Iteris inference API. Implements the contract in iteris_ui/src/api/contract.ts.

/predict serves the deployed Attention Res-U-Net baseline only; DuelingDDQN
and TD3 are served through the separate /infer entry point (see drl.py's
REGISTRY) and rejected here with a clear 400 if requested via /predict.
/compare mirrors /predict's baseline-only scope.
"""

import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import drl, inference, llm
from .drl import RegistryMiss
from .schemas import (
    CompareRequest,
    CompareResponse,
    CompareResult,
    HealthResponse,
    InferRequest,
    InferResponse,
    InterpretRequest,
    ModelRecord,
    PredictRequest,
    PredictResponse,
    SampleImage,
)

app = FastAPI(title='Iteris Inference API')

_origins_env = os.environ.get('CORS_ORIGINS', '*')
_origins = ['*'] if _origins_env == '*' else [o.strip() for o in _origins_env.split(',')]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

DEPLOYED_MODEL_ID = 'unet-baseline'

# Real Phase A evaluation numbers (2026-07-20 run) — diceCamus/diceBrisc are the
# per-dataset mean Dice (CAMUS across its 3 classes, BRISC's single tumor
# class); iou/hd are the mean across all 4 class rows (both datasets combined).
# Keep in sync with iteris_ui/src/content/models.yaml, which mirrors these
# exact figures for the frontend's mock/dev mode.
MODELS: list[ModelRecord] = [
    ModelRecord(
        id='unet-baseline',
        name='Attention U-Net',
        family='baseline',
        description=(
            'Attention Residual U-Net (Oktay et al. 2018 attention gates, ResNet-style '
            'encoder/decoder). Deployed baseline — serves as the reference segmentation '
            'against which DRL refinement is evaluated.'
        ),
        diceCamus=0.900,
        diceBrisc=0.870,
        iou=None,
        hd=6.793,
        deployed=True,
        selectable=True,
    ),
    ModelRecord(
        id='dueling-dqn', name='DuelingDDQN', family='discrete-drl',
        description=(
            'Dueling Double DQN with value/advantage stream decomposition and an '
            '8-direction discrete action space for boundary refinement.'
        ),
        diceCamus=0.896, diceBrisc=0.867, iou=0.788, hd=7.460, deployed=True, selectable=True,
    ),
    ModelRecord(
        id='td3', name='TD3', family='continuous-drl',
        description=(
            'Twin Delayed Deep Deterministic Policy Gradient with continuous vertex-'
            'displacement actions and clipped double-Q targets for stable policy learning.'
        ),
        diceCamus=0.885, diceBrisc=0.874, iou=0.784, hd=7.595, deployed=True, selectable=True,
    ),
]


@app.on_event('startup')
def _startup() -> None:
    inference.preload_models()


@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    import torch
    loaded = len(inference._MODEL_CACHE)
    return HealthResponse(
        status='ok' if loaded > 0 else 'degraded',
        modelsLoaded=loaded,
        gpuAvailable=torch.cuda.is_available(),
        datasetsAvailable=['camus', 'brisc'],
    )


@app.get('/models', response_model=list[ModelRecord])
def models() -> list[ModelRecord]:
    return MODELS


# thumbnailUrl/maskUrl are root-relative paths resolved against whatever
# origin renders them — i.e. the Vercel-hosted frontend, which serves these
# exact files from iteris_ui/public/samples/**. This Space never hosts the
# image bytes itself; it only needs to agree on the same relative paths.
# Keep in sync with iteris_ui/src/content/samples.yaml (same 6 real CAMUS/
# BRISC image+ground-truth-mask pairs, same ids/labels/bestDice figures) —
# sourced from the project's held-out test split (test-val-sets/holdout/),
# completely unseen during training.
#
# BRISC note: the meningioma case's filename token is 'me', which
# iteris/ingestion.py's TYPE_MAP doesn't recognise (it expects 'mn'), so the
# held-out manifest tags it "unknown" — verified against every manifest row
# that 'me' is meningioma, not a real non-tumour class.
SAMPLES: list[SampleImage] = [
    SampleImage(
        id='camus-a2c', thumbnailB64='',
        thumbnailUrl='/samples/camus/camus-a2c-image.png',
        maskUrl='/samples/camus/camus-a2c-mask.png',
        modality='ultrasound', anatomy='A2C (apical 2-chamber, ED)',
        difficulty='easy', bestDice=0.936, dataset='camus',
    ),
    SampleImage(
        id='camus-a4c', thumbnailB64='',
        thumbnailUrl='/samples/camus/camus-a4c-image.png',
        maskUrl='/samples/camus/camus-a4c-mask.png',
        modality='ultrasound', anatomy='A4C (apical 4-chamber, ED)',
        difficulty='medium', bestDice=0.894, dataset='camus',
    ),
    SampleImage(
        id='camus-4ch-es', thumbnailB64='',
        thumbnailUrl='/samples/camus/camus-4ch-es-image.png',
        maskUrl='/samples/camus/camus-4ch-es-mask.png',
        modality='ultrasound', anatomy='A4C (End-Systole frame)',
        difficulty='hard', bestDice=0.869, dataset='camus',
    ),
    SampleImage(
        id='brisc-glioma', thumbnailB64='',
        thumbnailUrl='/samples/brisc/brisc-glioma-image.jpg',
        maskUrl='/samples/brisc/brisc-glioma-mask.png',
        modality='mri', anatomy='Glioma',
        difficulty='hard', bestDice=0.867, dataset='brisc',
    ),
    SampleImage(
        id='brisc-meningioma', thumbnailB64='',
        thumbnailUrl='/samples/brisc/brisc-meningioma-image.jpg',
        maskUrl='/samples/brisc/brisc-meningioma-mask.png',
        modality='mri', anatomy='Meningioma',
        difficulty='medium', bestDice=0.870, dataset='brisc',
    ),
    SampleImage(
        id='brisc-pituitary', thumbnailB64='',
        thumbnailUrl='/samples/brisc/brisc-pituitary-image.jpg',
        maskUrl='/samples/brisc/brisc-pituitary-mask.png',
        modality='mri', anatomy='Pituitary Tumor',
        difficulty='easy', bestDice=0.874, dataset='brisc',
    ),
]


@app.get('/datasets/samples', response_model=list[SampleImage])
def samples() -> list[SampleImage]:
    return SAMPLES


@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if body.modelId != DEPLOYED_MODEL_ID:
        raise HTTPException(
            400,
            f"Model '{body.modelId}' is not served by /predict. Only "
            f"'{DEPLOYED_MODEL_ID}' (the Attention U-Net baseline) is — DRL "
            "models are routed through /infer instead.",
        )
    t0 = time.time()
    pred = inference.run_inference(body.dataset, body.imageB64)
    inference_ms = (time.time() - t0) * 1000

    structure_defs = inference.get_structure_defs(body.dataset, body.imageB64)
    masks = inference.build_masks(body.dataset, pred, structure_defs)
    metrics = inference.build_metrics(body.dataset, pred, body.gtMaskB64, structure_defs)

    return PredictResponse(
        sessionId=str(uuid.uuid4()),
        modelId=body.modelId,
        dataset=body.dataset,
        masks=masks,
        metrics=metrics,
        preprocessingMs=0.0,
        inferenceMs=round(inference_ms, 1),
        imageWidth=inference.IMAGE_SIZE,
        imageHeight=inference.IMAGE_SIZE,
    )


@app.post('/compare', response_model=CompareResponse)
def compare(body: CompareRequest) -> CompareResponse:
    results: list[CompareResult] = []
    for model_id in body.modelIds:
        if model_id != DEPLOYED_MODEL_ID:
            continue  # skip undeployed models rather than erroring the whole request
        pred = inference.run_inference(body.dataset, body.imageB64)
        structure_defs = inference.get_structure_defs(body.dataset, body.imageB64)
        results.append(CompareResult(
            modelId=model_id,
            masks=inference.build_masks(body.dataset, pred, structure_defs),
            metrics=inference.build_metrics(body.dataset, pred, None, structure_defs),
        ))
    return CompareResponse(results=results)


@app.post('/interpret')
def interpret(body: InterpretRequest) -> StreamingResponse:
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise HTTPException(501, 'ANTHROPIC_API_KEY is not configured on this Space.')
    return StreamingResponse(llm.stream_interpretation(body), media_type='text/plain')


@app.post('/infer', response_model=InferResponse)
def infer(body: InferRequest) -> InferResponse:
    """
    Generic inference entry point keyed by (dataset, model_family, algo, regime)
    — the DRL counterpart to /predict (which only ever serves DEPLOYED_MODEL_ID).
    Currently only camus/drl/duelingddqn/low has a registered+configured
    checkpoint; every other combination 404s with the missing key so the
    frontend can grey out that option instead of surfacing a failed request.
    """
    try:
        result = drl.infer(
            dataset=body.dataset,
            model_family=body.modelFamily,
            algo=body.algo,
            regime=body.regime,
            image_b64=body.imageB64,
            gt_b64=body.gtMaskB64,
        )
    except RegistryMiss as exc:
        raise HTTPException(404, {
            'error': 'not_registered',
            'key': {
                'dataset': body.dataset,
                'modelFamily': body.modelFamily,
                'algo': body.algo,
                'regime': body.regime,
            },
            'detail': str(exc),
        })
    except Exception as exc:  # noqa: BLE001 — surface the real cause instead of a bare 500
        import traceback
        raise HTTPException(500, {
            'error': 'infer_failed',
            'exception': type(exc).__name__,
            'detail': str(exc),
            'traceback': traceback.format_exc()[-4000:],
        })
    return InferResponse(
        dataset=body.dataset, algo=body.algo, regime=body.regime, **result,
    )
