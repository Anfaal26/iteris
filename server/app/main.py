"""
Iteris inference API. Implements the contract in
iteris_ui/src/api/contract.ts against the trained Attention Res-U-Net
checkpoints only — DRL agents (dqn/ddqn/dueling-dqn/ddpg) are not trained yet,
so /predict and /compare reject those modelIds with a clear 400 until the
checkpoints exist.
"""

import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import inference, llm
from .schemas import (
    CompareRequest,
    CompareResponse,
    CompareResult,
    HealthResponse,
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

MODELS: list[ModelRecord] = [
    ModelRecord(
        id='unet-baseline',
        name='Attention U-Net',
        family='baseline',
        description=(
            'Attention Residual U-Net (Oktay et al. 2018 attention gates, ResNet-style '
            'encoder/decoder). Currently the only deployed checkpoint — DRL refinement '
            'agents below are trained but not yet evaluated/deployed.'
        ),
        diceCamus=None,
        diceBrisc=None,
        iou=None,
        hd=None,
        deployed=True,
        selectable=True,
    ),
    ModelRecord(
        id='dqn', name='DQN', family='discrete-drl',
        description='Double Deep Q-Network contour refinement. Training in progress — not yet deployed.',
        diceCamus=None, diceBrisc=None, iou=None, hd=None, deployed=False, selectable=False,
    ),
    ModelRecord(
        id='ddqn', name='DDQN', family='discrete-drl',
        description='Double DQN with separate online/target networks. Training in progress — not yet deployed.',
        diceCamus=None, diceBrisc=None, iou=None, hd=None, deployed=False, selectable=False,
    ),
    ModelRecord(
        id='dueling-dqn', name='Dueling DQN', family='discrete-drl',
        description='Dueling Double DQN. Training in progress — not yet deployed.',
        diceCamus=None, diceBrisc=None, iou=None, hd=None, deployed=False, selectable=False,
    ),
    ModelRecord(
        id='ddpg', name='DDPG', family='continuous-drl',
        description='Continuous-action actor-critic contour refinement. Training in progress — not yet deployed.',
        diceCamus=None, diceBrisc=None, iou=None, hd=None, deployed=False, selectable=False,
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


@app.get('/datasets/samples', response_model=list[SampleImage])
def samples() -> list[SampleImage]:
    # No curated sample thumbnails wired up yet — upload path is the supported
    # flow for now. Populate this once real CAMUS/BRISC sample PNGs + GT masks
    # are chosen and base64-embedded (see iteris_ui/src/content/samples.yaml).
    return []


@app.post('/predict', response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if body.modelId != DEPLOYED_MODEL_ID:
        raise HTTPException(
            400,
            f"Model '{body.modelId}' is not deployed yet. Only '{DEPLOYED_MODEL_ID}' "
            '(the trained Attention U-Net) is currently served.',
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
